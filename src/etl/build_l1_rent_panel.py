from __future__ import annotations

import argparse
import csv
import hashlib
import http.client
import json
import logging
import platform
import shlex
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


getcontext().prec = 50

PROTOCOL_START = date(2022, 1, 1)
DENCUN_DATE = date(2024, 3, 13)
# Historical Blockscout txlist gasUsed is materially under-reported on pre-Dencun Optimism
# batcher submissions. Use receipts to meter that calldata scope while leaving unaffected
# calldata windows on the lighter txlist path.
CALDATA_RECEIPT_METERING_ROLLUPS = frozenset({"optimism"})
WEI_PER_ETH = Decimal("1000000000000000000")
WEI_PER_GWEI = Decimal("1000000000")

BLOCKSCOUT_TXLIST_URL = "https://eth.blockscout.com/api"
BLOCKSCOUT_RPC_URL = "https://eth.blockscout.com/api/eth-rpc"
ETH_FALLBACK_RPC_URL = "https://ethereum-rpc.publicnode.com"
BLOBSCAN_TX_URL = "https://api.blobscan.com/transactions"

BLOCKSCOUT_HEADERS = {
    "Accept": "application/json,text/plain,*/*",
    "User-Agent": "Mozilla/5.0",
}
BROWSER_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "User-Agent": "Mozilla/5.0",
}
BLOBSCAN_HEADERS = {
    "Accept": "application/json,text/plain,*/*",
    "User-Agent": "Mozilla/5.0",
}

BLOCKSCOUT_TX_PAGE_SIZE = 1000
BLOCKSCOUT_MIN_PAGE_SIZE = 250
BLOBSCAN_TX_PAGE_SIZE = 500
BLOBSCAN_MIN_PAGE_SIZE = 100
# Blobscan occasionally returns transient 502/503s on page 1 for otherwise valid windows.
# Keep a brief cooldown, but split/fallback quickly when instability persists so long resume
# runs do not spend minutes sleeping on one pathological month slice.
BLOBSCAN_INSTABILITY_RETRY_DELAY_SECONDS = 10.0
BLOBSCAN_INSTABILITY_RETRY_ROUNDS = 1
# Public Blobscan occasionally returns transient 502/503s even for exact one-second windows
# that later succeed as empty results. Give terminal windows a few slower retries before
# classifying the source as broken.
BLOBSCAN_TERMINAL_WINDOW_RETRY_DELAY_SECONDS = 30.0
BLOBSCAN_TERMINAL_WINDOW_RETRY_ROUNDS = 2
# The public Ethereum Blockscout eth-rpc endpoint currently rejects JSON-RPC batches larger
# than 5 with HTTP 413 ("Payload Too Large. Max batch size is 5"). Keep the default aligned
# to the live provider limit so the receipt/base-fee enrichment phase can resume safely.
RPC_BATCH_SIZE = 5
BIGQUERY_PROJECT_ID = "l2-l1-causal-analysis"
BIGQUERY_PUBLIC_ETHEREUM_TRANSACTIONS_TABLE = "bigquery-public-data.crypto_ethereum.transactions"
BIGQUERY_PUBLIC_ETHEREUM_BLOCKS_TABLE = "bigquery-public-data.crypto_ethereum.blocks"
BIGQUERY_BLOCKSCOUT_TRANSACTIONS_TABLE = "bigquery-public-data.blockchain_analytics_ethereum_mainnet_us.transactions"
BIGQUERY_BLOCKSCOUT_RECEIPTS_TABLE = "bigquery-public-data.blockchain_analytics_ethereum_mainnet_us.receipts"
BIGQUERY_BLOCKSCOUT_QUERY_MAX_ROWS = 1000000
BIGQUERY_RECEIPT_QUERY_CHUNK_SIZE = 10000
BIGQUERY_BLOCK_BASE_FEE_QUERY_CHUNK_SIZE = 20000
BIGQUERY_BLOCK_BASE_FEE_RANGE_SIZE = 200000
BIGQUERY_BLOCK_BASE_FEE_RANGE_DENSITY_THRESHOLD = 0.25
LOOKUP_DB_SELECT_CHUNK_SIZE = 500
BIGQUERY_RECEIPT_PROGRESS_LOG_INTERVAL = 5
RECEIPT_FALLBACK_PROGRESS_LOG_INTERVAL = 1000
PARTITION_CHECKPOINT_SCHEMA_VERSION = 1
PARTITION_CHECKPOINT_COMPAT_VERSION = 4

PANEL_HEADERS = [
    "date_utc",
    "rollup_id",
    "l2_fees_eth",
    "rent_paid_eth",
    "profit_eth",
    "txcount",
]
DECOMP_HEADERS = [
    "date_utc",
    "l1_base_fee_burn_eth",
    "l1_blob_fee_burn_eth",
    "l1_priority_fee_eth",
    "l1_total_rent_eth",
    "l1_blob_gas_used",
    "l1_calldata_gas_used",
    "l1_blob_base_fee_gwei",
]
COMPONENT_HEADERS = [
    "date_utc",
    "rollup_id",
    "batch_submissions_eth",
    "proof_submissions_eth",
    "state_updates_eth",
    "execution_base_fee_burn_eth",
    "execution_priority_fee_eth",
    "blob_fee_burn_eth",
    "rent_paid_eth",
]
SAMPLE_ROLLUPS = ("arbitrum", "base", "optimism")
SAMPLE_DATES = ("2024-03-13", "2024-03-14", "2024-03-15")
ROLLUPS_WITHOUT_BATCHER_ADDRESSES = {"scroll"}
BLOBSCAN_ROLLUP_ALIASES = {
    "world": "worldchain",
    "zksync": "zksync_era",
}
ROLLUP_SUBTYPE_TO_COMPONENT_FIELD = {
    "batchSubmissions": "batch_submissions_wei",
    "proofSubmissions": "proof_submissions_wei",
    "stateUpdates": "state_updates_wei",
}
CANONICAL_EXCLUDED_SUBTYPES_BY_ROLLUP = {
    # T051 locks Starknet canonical rent to the direct-exclusive state-update surface
    # until a reviewed shared-SHARP allocation model exists.
    "starknet": frozenset({"batchSubmissions", "proofSubmissions"}),
}


@dataclass(frozen=True)
class RegistryRollup:
    rollup_id: str
    start_date_utc: date
    end_date_utc: date | None
    batcher_addresses: tuple[str, ...]
    evidence_url: str


@dataclass(frozen=True)
class BlockscoutTx:
    hash: str
    rollup_id: str
    subtype: str
    address: str
    block_number: int
    timestamp_utc: datetime
    to_address: str | None
    method_id: str | None
    gas_price_wei: int
    gas_used: int
    value_wei: int
    txreceipt_status: str | None


@dataclass(frozen=True)
class BlobscanTx:
    hash: str
    rollup_id: str
    subtype: str
    block_number: int
    timestamp_utc: datetime
    from_address: str
    to_address: str | None
    blob_gas_used: int
    blob_gas_price_wei: int
    blob_as_calldata_gas_used: int


@dataclass(frozen=True)
class ReceiptFields:
    hash: str
    block_number: int
    from_address: str
    to_address: str | None
    gas_used: int
    effective_gas_price_wei: int
    blob_gas_used: int
    blob_gas_price_wei: int


@dataclass(frozen=True)
class FetchResult:
    payload: Any
    raw_bytes: bytes
    fetched_at_utc: str


@dataclass(frozen=True)
class PartitionCheckpoint:
    checkpoint_path: Path
    calldata_candidate_txs_count: int
    excluded_blob_overlap_txs: int
    calldata_txs_count: int
    blob_txs_count: int
    request_log: list[dict[str, Any]]


@dataclass(frozen=True)
class ExactBlobscanScopeCache:
    rows: list[BlobscanTx]
    page_count: int
    page_size: int
    last_page_row_count: int
    max_positive_total_transactions: int | None
    contiguous_pages: bool


@dataclass(frozen=True)
class TrackedFunctionCall:
    rollup_id: str
    subtype: str
    address: str
    selector: str
    signature: str
    since_timestamp: int
    until_timestamp: int | None


TRACKED_TRANSACTION_SUBTYPES = ("batchSubmissions", "proofSubmissions", "stateUpdates")
LEGACY_TRACKED_CALLS_BY_ROLLUP: dict[str, tuple[TrackedFunctionCall, ...]] = {
    "arbitrum": (
        TrackedFunctionCall(
            rollup_id="arbitrum",
            subtype="batchSubmissions",
            # Arbitrum Classic posted sequencer batches to the legacy mainnet
            # Sequencer Inbox before Nitro. The current L2BEAT feed only covers
            # the Nitro-era SequencerInbox, so keep the evidence-backed classic
            # hook here to cover the entire 2022-01-01..2022-08-30 interval.
            address="0x4c6f947ae67f572afa4ae0730947de7c874f95ef",
            selector="0x8a2df18d",
            signature="function addSequencerL2BatchFromOriginWithGasRefunder(bytes transactions, uint256[] lengths, uint256[] sectionsMetadata, bytes32 afterAcc, address gasRefunder)",
            since_timestamp=1640995780,
            until_timestamp=1661903988,
        ),
        TrackedFunctionCall(
            rollup_id="arbitrum",
            subtype="batchSubmissions",
            # Nitro launch used a short-lived SequencerInbox selector on the
            # modern inbox contract before the long-lived 0x8f111f3c path took
            # over on 2022-09-19. Keep the legacy selector explicit so the
            # launch-era 2022-08-31..2022-09-19 rent is not silently dropped.
            address="0x1c479675ad559dc151f6ec7ed3fbf8cee79582b6",
            selector="0x6f12b0c9",
            signature="legacy launch-era Nitro SequencerInbox batch submission selector 0x6f12b0c9",
            since_timestamp=1661964203,
            until_timestamp=1663615751,
        ),
    ),
    "optimism": (
        TrackedFunctionCall(
            rollup_id="optimism",
            subtype="stateUpdates",
            # Legacy OP Mainnet state roots were posted to the State Commitment
            # Chain before Bedrock. L2BEAT's current trackedTransactions feed no
            # longer exposes this historical hook, so keep the evidence-backed
            # selector here to avoid silently undercounting OP L1 rent.
            address="0xbe5dab4a2e9cd0f27300db4ab94bee3a233aeb19",
            selector="0x8ca5cbb9",
            signature="function appendStateBatch(bytes32[] _batch, uint256 _shouldStartAtElement)",
            since_timestamp=1636588800,
            until_timestamp=1686095999,
        ),
    ),
    "zksync_era": (
        TrackedFunctionCall(
            rollup_id="zksync_era",
            subtype="batchSubmissions",
            # The pre-Boojum Era main contract on 0x3db5... kept the legacy
            # batch commit selector 0x0c4dd810 off the current L2BEAT tracked
            # transaction feed. BigQuery shows it active from 2023-03-24
            # through 2023-12-07 and it is a dominant missing cost family in
            # the worst 2023 reconciliation months.
            address="0x3db52ce065f728011ac6732222270b3f2360d919",
            selector="0x0c4dd810",
            signature="legacy Era main-contract batch submission selector 0x0c4dd810",
            since_timestamp=1679654255,
            until_timestamp=1701700367,
        ),
        TrackedFunctionCall(
            rollup_id="zksync_era",
            subtype="batchSubmissions",
            # The shared-bridge commit selector kept posting on-chain well past
            # the current L2BEAT until bound. Extend it to the observed last
            # on-chain use instead of truncating live rent-bearing history.
            address="0x5d8ba173dc6c3c90c8f7c04c9288bef5fdbad06e",
            selector="0x6edd4f12",
            signature="function commitBatchesSharedBridge(uint256 _chainId, (uint64 batchNumber, bytes32 batchHash, uint64 indexRepeatedStorageChanges, uint256 numberOfLayer1Txs, bytes32 priorityOperationsHash, bytes32 l2LogsTreeRoot, uint256 timestamp, bytes32 commitment), (uint64 batchNumber, uint64 timestamp, uint64 indexRepeatedStorageChanges, bytes32 newStateRoot, uint256 numberOfLayer1Txs, bytes32 priorityOperationsHash, bytes32 bootloaderHeapInitialContentsHash, bytes32 eventsQueueStateHash, bytes systemLogs, bytes pubdataCommitments)[] _newBatchesData)",
            since_timestamp=1722410364,
            until_timestamp=1743088391,
        ),
        TrackedFunctionCall(
            rollup_id="zksync_era",
            subtype="proofSubmissions",
            address="0x5d8ba173dc6c3c90c8f7c04c9288bef5fdbad06e",
            selector="0xc37533bb",
            signature="function proveBatchesSharedBridge(uint256 _chainId, (uint64 batchNumber, bytes32 batchHash, uint64 indexRepeatedStorageChanges, uint256 numberOfLayer1Txs, bytes32 priorityOperationsHash, bytes32 l2LogsTreeRoot, uint256 timestamp, bytes32 commitment), (uint64 batchNumber, bytes32 batchHash, uint64 indexRepeatedStorageChanges, uint256 numberOfLayer1Txs, bytes32 priorityOperationsHash, bytes32 l2LogsTreeRoot, uint256 timestamp, bytes32 commitment)[], (uint256[] recursiveAggregationInput, uint256[] serializedProof))",
            since_timestamp=1722410364,
            until_timestamp=1743094943,
        ),
        TrackedFunctionCall(
            rollup_id="zksync_era",
            subtype="stateUpdates",
            address="0x5d8ba173dc6c3c90c8f7c04c9288bef5fdbad06e",
            selector="0x6f497ac6",
            signature="function executeBatchesSharedBridge(uint256 _chainId, (uint64 batchNumber, bytes32 batchHash, uint64 indexRepeatedStorageChanges, uint256 numberOfLayer1Txs, bytes32 priorityOperationsHash, bytes32 l2LogsTreeRoot, uint256 timestamp, bytes32 commitment)[] _newBatchesData)",
            since_timestamp=1722410364,
            until_timestamp=1743088235,
        ),
    ),
    "taiko": (
        TrackedFunctionCall(
            rollup_id="taiko",
            subtype="stateUpdates",
            # Taiko's historical proving traffic did not stay on TaikoL1/Inbox.
            # Official Taiko docs and deployment logs identify 0x68d3... as the
            # labprover / ProverSet proxy, and BigQuery shows the current
            # L2BEAT tracked feed misses this older proving surface entirely.
            # Keep the observed 2024-05-25..2024-11-07 proveBlock window
            # explicit so canonical Taiko attribution covers the official
            # operator path instead of only the later Inbox route.
            address="0x68d30f47f19c07bccef4ac7fae2dc12fca3e0dc9",
            selector="0x10d008bd",
            signature="historical Taiko labprover proveBlock selector 0x10d008bd",
            since_timestamp=1716625487,
            until_timestamp=1730973167,
        ),
        TrackedFunctionCall(
            rollup_id="taiko",
            subtype="batchSubmissions",
            # BigQuery shows the official labprover / ProverSet proxy on
            # 0x68d3... also handled a large historical share of Taiko block
            # proposals before the later Inbox-only surface. L2BEAT's current
            # trackedTransactions feed only carries the 0x06a9... Inbox path, so
            # preserve the observed 2024-06-08..2024-11-07 proposeBlock window
            # here instead of silently dropping this contract regime.
            address="0x68d30f47f19c07bccef4ac7fae2dc12fca3e0dc9",
            selector="0xef16e845",
            signature="historical Taiko labprover proposeBlock selector 0xef16e845",
            since_timestamp=1717845743,
            until_timestamp=1730973071,
        ),
        TrackedFunctionCall(
            rollup_id="taiko",
            subtype="batchSubmissions",
            # After Ontake, the same historical labprover / ProverSet proxy kept
            # handling Taiko proposal traffic via proposeBlocksV2 through the
            # 2025-01 tail. Without this explicit supplement, canonical Taiko
            # misses the dominant non-Inbox proposal surface in the remaining
            # T050 gap months.
            address="0x68d30f47f19c07bccef4ac7fae2dc12fca3e0dc9",
            selector="0x0c8f4a10",
            signature="historical Taiko labprover proposeBlocksV2 selector 0x0c8f4a10",
            since_timestamp=1730973119,
            until_timestamp=1738367975,
        ),
        TrackedFunctionCall(
            rollup_id="taiko",
            subtype="stateUpdates",
            # The historical labprover / ProverSet proxy also carried Taiko's
            # proveBlocks flow during the same 2024-11..2025-01 regime. Keep
            # the observed window explicit so canonical attribution follows the
            # real proving surface rather than assuming every proof hit Inbox.
            address="0x68d30f47f19c07bccef4ac7fae2dc12fca3e0dc9",
            selector="0x440b6e18",
            signature="historical Taiko labprover proveBlocks selector 0x440b6e18",
            since_timestamp=1730973227,
            until_timestamp=1738367759,
        ),
        TrackedFunctionCall(
            rollup_id="taiko",
            subtype="stateUpdates",
            # The short-lived verifyBlocks selector on Inbox disappeared from
            # the current L2BEAT tracked feed, but it is still observable on
            # chain during the 2024-07-27..2024-09-08 transition window.
            address="0x06a9ab27c7e2255df1815e6cc0168d7755feb19a",
            selector="0x8778209d",
            signature="historical Taiko verifyBlocks selector 0x8778209d",
            since_timestamp=1722088679,
            until_timestamp=1725839183,
        ),
        TrackedFunctionCall(
            rollup_id="taiko",
            subtype="batchSubmissions",
            # The current L2BEAT tracked transaction feed skips the short-lived
            # proposer selector 0xe4882785 even though the local sender snapshot
            # shows it posting continuously from 2025-03-18 through 2025-05-21.
            # Keep the evidence-backed contract/selector window explicit so the
            # transition between proposeBlocksV2 and later batch methods is not
            # silently dropped from canonical rent.
            address="0x68d30f47f19c07bccef4ac7fae2dc12fca3e0dc9",
            selector="0xe4882785",
            signature="Taiko proposer selector 0xe4882785 (proposeBlocksV2Conditionally)",
            since_timestamp=1742294171,
            until_timestamp=1747823663,
        ),
    ),
}


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="build_l1_rent_panel.py")
    parser.add_argument("--run-date", required=True, type=parse_date, help="UTC as-of date (YYYY-MM-DD)")
    parser.add_argument("--retries", type=int, default=4, help="Retry attempts for network fetches")
    parser.add_argument("--timeout-seconds", type=float, default=45.0, help="Per-request timeout in seconds")
    parser.add_argument(
        "--blockscout-page-size",
        type=int,
        default=BLOCKSCOUT_TX_PAGE_SIZE,
        help="Rows per Blockscout txlist page",
    )
    parser.add_argument(
        "--blobscan-page-size",
        type=int,
        default=BLOBSCAN_TX_PAGE_SIZE,
        help="Rows per Blobscan transaction page",
    )
    parser.add_argument(
        "--rpc-batch-size",
        type=int,
        default=RPC_BATCH_SIZE,
        help="JSON-RPC requests per batch",
    )
    parser.add_argument(
        "--resume-manifested-run",
        action="store_true",
        help="Reuse an existing raw snapshot/manifests for the same run date after a deterministic ETL repair",
    )
    return parser.parse_args(argv)


def parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid date {value!r}; expected YYYY-MM-DD") from exc


def parse_datetime(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SystemExit(f"invalid timestamp in source response: {value!r}") from exc
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def git_sha(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def command_string(args: argparse.Namespace) -> str:
    command = ["python", "src/etl/build_l1_rent_panel.py", "--run-date", args.run_date.isoformat()]
    if args.retries != 4:
        command.extend(["--retries", str(args.retries)])
    if args.timeout_seconds != 45.0:
        command.extend(["--timeout-seconds", str(args.timeout_seconds)])
    if args.blockscout_page_size != BLOCKSCOUT_TX_PAGE_SIZE:
        command.extend(["--blockscout-page-size", str(args.blockscout_page_size)])
    if args.blobscan_page_size != BLOBSCAN_TX_PAGE_SIZE:
        command.extend(["--blobscan-page-size", str(args.blobscan_page_size)])
    if args.rpc_batch_size != RPC_BATCH_SIZE:
        command.extend(["--rpc-batch-size", str(args.rpc_batch_size)])
    if args.resume_manifested_run:
        command.append("--resume-manifested-run")
    return " ".join(shlex.quote(token) for token in command)


def prepare_snapshot_dir(path: Path, *, raw_manifest_path: Path) -> None:
    if path.exists():
        if raw_manifest_path.exists():
            raise SystemExit(
                f"append-only raw snapshot already exists and has a raw manifest: {path}. "
                "Choose a new --run-date instead of overwriting prior pulls."
            )
        if not path.is_dir():
            raise SystemExit(f"raw snapshot path exists but is not a directory: {path}")
        logging.info("Resuming incomplete raw snapshot without overwriting existing files: %s", path)
        return
    path.mkdir(parents=True, exist_ok=False)


def ensure_new_manifest(path: Path, *, label: str) -> None:
    if path.exists():
        raise SystemExit(
            f"{label} already exists for this run date: {path}. "
            "Manifests are append-only; do not overwrite prior provenance."
        )


def parse_blockscout_tx_record(
    row: dict[str, Any],
    *,
    rollup_id: str,
    address: str,
    default_subtype: str = "batchSubmissions",
) -> BlockscoutTx:
    raw_to = row.get("to_address")
    to_address = str(raw_to).strip().lower() if isinstance(raw_to, str) and raw_to.strip() else None
    raw_method_id = row.get("method_id")
    method_id = str(raw_method_id).strip().lower() if isinstance(raw_method_id, str) and raw_method_id.strip() else None
    try:
        parsed = BlockscoutTx(
            hash=str(row["hash"]).strip().lower(),
            rollup_id=str(row["rollup_id"]).strip(),
            subtype=str(row.get("subtype", default_subtype)).strip(),
            address=str(row["address"]).strip().lower(),
            block_number=int(row["block_number"]),
            timestamp_utc=parse_datetime(str(row["timestamp_utc"])),
            to_address=to_address,
            method_id=method_id,
            gas_price_wei=int(str(row["gas_price_wei"])),
            gas_used=int(row["gas_used"]),
            value_wei=int(str(row.get("value_wei", "0"))),
            txreceipt_status=str(row["txreceipt_status"]) if row.get("txreceipt_status") is not None else None,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"stored Blockscout tx record is malformed in resume path: {row!r}") from exc

    if parsed.rollup_id != rollup_id or parsed.address != address:
        raise SystemExit(
            f"stored Blockscout tx record does not match requested resume scope "
            f"({rollup_id}/{address}): {row!r}"
        )
    if parsed.subtype not in ROLLUP_SUBTYPE_TO_COMPONENT_FIELD:
        raise SystemExit(f"stored Blockscout tx record has unknown subtype: {row!r}")
    return parsed


def load_existing_blockscout_page(
    path: Path,
    *,
    rollup_id: str,
    address: str,
    default_subtype: str = "batchSubmissions",
) -> tuple[list[BlockscoutTx], int, int]:
    payload = read_json(path)
    rows = payload.get("transactions")
    if not isinstance(rows, list):
        raise SystemExit(f"stored Blockscout page is malformed: {path}")
    stored_page_size = payload.get("page_size")
    if stored_page_size is None:
        stored_page_size = len(rows)
    else:
        try:
            stored_page_size = int(stored_page_size)
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"stored Blockscout page has malformed page_size: {path}") from exc
    if stored_page_size <= 0:
        raise SystemExit(f"stored Blockscout page has non-positive page_size: {path}")
    stored_result_count = payload.get("result_count")
    if stored_result_count is None:
        stored_result_count = len(rows)
    else:
        try:
            stored_result_count = int(stored_result_count)
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"stored Blockscout page has malformed result_count: {path}") from exc
    if stored_result_count < 0:
        raise SystemExit(f"stored Blockscout page has negative result_count: {path}")
    return (
        [
            parse_blockscout_tx_record(
                row,
                rollup_id=rollup_id,
                address=address,
                default_subtype=default_subtype,
            )
            for row in rows
            if isinstance(row, dict)
        ],
        stored_page_size,
        stored_result_count,
    )


def requires_calldata_receipt_metering(tx: BlockscoutTx) -> bool:
    return tx.rollup_id in CALDATA_RECEIPT_METERING_ROLLUPS and tx.timestamp_utc.date() < DENCUN_DATE


def receipt_hash_scope_selects(*, include_calldata: bool) -> list[str]:
    selects: list[str] = []
    if include_calldata:
        rollup_literals = ", ".join(f"'{rollup_id}'" for rollup_id in sorted(CALDATA_RECEIPT_METERING_ROLLUPS))
        selects.append(
            "SELECT c.hash "
            "FROM calldata_txs c "
            f"WHERE c.rollup_id IN ({rollup_literals}) "
            f"AND substr(c.timestamp_utc, 1, 10) < '{DENCUN_DATE.isoformat()}'"
        )
    selects.append("SELECT hash FROM blob_txs")
    return selects


def blockscout_page_dir(
    *,
    snapshot_dir: Path,
    path_prefix: str,
    rollup_id: str,
    address: str,
    scope_id: str | None,
) -> Path:
    page_dir = snapshot_dir / "blockscout" / path_prefix / rollup_id / address
    if scope_id is not None:
        page_dir = page_dir / scope_id
    return page_dir


def overlapping_blockscout_cache_exists(
    *,
    page_dir: Path,
    window_start_dt: datetime,
    window_end_exclusive_dt: datetime,
    window_id: str,
) -> bool:
    if not page_dir.exists():
        return False
    for candidate_path in sorted(page_dir.glob("*_page-*.json")):
        if candidate_path.name.startswith(f"{window_id}_page-"):
            continue
        payload = read_json(candidate_path)
        raw_start = payload.get("window_start_utc")
        raw_end = payload.get("window_end_exclusive_utc")
        if not isinstance(raw_start, str) or not isinstance(raw_end, str):
            continue
        try:
            candidate_start_dt = parse_datetime(raw_start)
            candidate_end_exclusive_dt = parse_datetime(raw_end)
        except SystemExit:
            continue
        if candidate_end_exclusive_dt <= window_start_dt or candidate_start_dt >= window_end_exclusive_dt:
            continue
        return True
    return False


def maybe_reuse_blockscout_window_from_overlapping_cache(
    *,
    page_dir: Path,
    rollup_id: str,
    address: str,
    window_start_dt: datetime,
    window_end_exclusive_dt: datetime,
    window_id: str,
    request_log: list[dict[str, Any]],
    address_role: str,
    method_selectors: tuple[str, ...] | None,
    scope_id: str | None,
    subtype: str = "batchSubmissions",
) -> list[BlockscoutTx] | None:
    if not page_dir.exists():
        return None

    groups: dict[tuple[datetime, datetime], list[tuple[int, Path]]] = {}
    for candidate_path in sorted(page_dir.glob("*_page-*.json")):
        if candidate_path.name.startswith(f"{window_id}_page-"):
            continue
        payload = read_json(candidate_path)
        raw_start = payload.get("window_start_utc")
        raw_end = payload.get("window_end_exclusive_utc")
        if not isinstance(raw_start, str) or not isinstance(raw_end, str):
            continue
        try:
            candidate_start_dt = parse_datetime(raw_start)
            candidate_end_exclusive_dt = parse_datetime(raw_end)
        except SystemExit:
            continue
        if candidate_start_dt > window_start_dt or candidate_end_exclusive_dt < window_end_exclusive_dt:
            continue
        try:
            page_number = int(candidate_path.stem.rsplit("_page-", 1)[1])
        except (IndexError, ValueError):
            continue
        groups.setdefault((candidate_start_dt, candidate_end_exclusive_dt), []).append((page_number, candidate_path))

    candidate_groups = sorted(
        groups.items(),
        key=lambda item: (
            (item[0][1] - item[0][0]).total_seconds(),
            item[0][0],
        ),
    )
    for (candidate_start_dt, candidate_end_exclusive_dt), page_entries in candidate_groups:
        sorted_entries = sorted(page_entries, key=lambda entry: entry[0])
        expected_page = 1
        stored_page_size: int | None = None
        stored_result_count: int | None = None
        candidate_rows: list[BlockscoutTx] = []
        reusable = True
        for page_number, candidate_path in sorted_entries:
            if page_number != expected_page:
                reusable = False
                break
            rows, page_size, result_count = load_existing_blockscout_page(
                candidate_path,
                rollup_id=rollup_id,
                address=address,
                default_subtype=subtype,
            )
            if stored_page_size is None:
                stored_page_size = page_size
            elif page_size != stored_page_size:
                reusable = False
                break
            request_log.append(
                {
                    "source": "blockscout_txlist",
                    "rollup_id": rollup_id,
                    "address": address,
                    "filter_by": address_role,
                    "method_selectors": list(method_selectors) if method_selectors else None,
                    "scope_id": scope_id,
                    "window_start_utc": candidate_start_dt.isoformat(),
                    "window_end_exclusive_utc": candidate_end_exclusive_dt.isoformat(),
                    "page": page_number,
                    "page_size": stored_page_size,
                    "relative_path": str(candidate_path.relative_to(repo_root())),
                    "fetched_at_utc": None,
                    "reused_existing": True,
                    "reused_via": "overlapping_complete_window",
                }
            )
            candidate_rows.extend(rows)
            stored_result_count = result_count
            expected_page += 1
        if not reusable or stored_page_size is None or stored_result_count is None:
            continue
        if stored_result_count >= stored_page_size:
            continue

        filtered_rows: list[BlockscoutTx] = []
        seen_hashes: set[str] = set()
        for row in candidate_rows:
            if row.timestamp_utc < window_start_dt or row.timestamp_utc >= window_end_exclusive_dt:
                continue
            if row.hash in seen_hashes:
                continue
            seen_hashes.add(row.hash)
            filtered_rows.append(row)

        logging.info(
            "Reused %s Blockscout rows for %s/%s within %s..%s from enclosing cached window %s..%s",
            len(filtered_rows),
            rollup_id,
            address,
            window_start_dt.isoformat(),
            window_end_exclusive_dt.isoformat(),
            candidate_start_dt.isoformat(),
            candidate_end_exclusive_dt.isoformat(),
        )
        return filtered_rows
    return None


def backfill_blockscout_window_from_bigquery(
    *,
    snapshot_dir: Path,
    rollup_id: str,
    address: str,
    window_start_dt: datetime,
    window_end_exclusive_dt: datetime,
    page_size: int,
    request_log: list[dict[str, Any]],
    address_role: str,
    method_selectors: tuple[str, ...] | None,
    path_prefix: str,
    scope_id: str | None,
    subtype: str = "batchSubmissions",
) -> list[BlockscoutTx] | None:
    address_field = "from_address" if address_role == "from" else "to_address"
    selector_clause = ""
    if method_selectors:
        selector_literals = ", ".join(f"'{selector.lower()}'" for selector in sorted(set(method_selectors)))
        selector_clause = f"  AND LOWER(SUBSTR(t.input, 1, 10)) IN ({selector_literals})\n"
    where_clause = (
        f"WHERE t.block_timestamp >= TIMESTAMP('{window_start_dt.isoformat()}')\n"
        f"  AND t.block_timestamp < TIMESTAMP('{window_end_exclusive_dt.isoformat()}')\n"
        f"  AND LOWER(t.{address_field}) = '{address}'\n"
        f"{selector_clause}"
    )
    count_query = (
        "SELECT COUNT(DISTINCT t.transaction_hash) AS row_count\n"
        f"FROM `{BIGQUERY_BLOCKSCOUT_TRANSACTIONS_TABLE}` t\n"
        f"{where_clause}"
    )
    try:
        count_result = subprocess.run(
            [
                "bq",
                "query",
                f"--project_id={BIGQUERY_PROJECT_ID}",
                "--use_legacy_sql=false",
                "--format=json",
                "--max_rows=1",
            ],
            input=count_query,
            text=True,
            capture_output=True,
            check=True,
        )
    except FileNotFoundError:
        logging.warning(
            "Skipping BigQuery Blockscout backfill because the bq CLI is unavailable; falling back to live Blockscout for %s/%s within %s..%s",
            rollup_id,
            address,
            window_start_dt.isoformat(),
            window_end_exclusive_dt.isoformat(),
        )
        return None
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        logging.warning(
            "BigQuery Blockscout count query failed for %s/%s within %s..%s via project %s: %s; falling back to live Blockscout",
            rollup_id,
            address,
            window_start_dt.isoformat(),
            window_end_exclusive_dt.isoformat(),
            BIGQUERY_PROJECT_ID,
            stderr[:400] if stderr else exc,
        )
        return None
    try:
        count_payload = json.loads(count_result.stdout)
        expected_rows = int(count_payload[0]["row_count"]) if count_payload else 0
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        logging.warning(
            "BigQuery Blockscout count query returned malformed payload for %s/%s within %s..%s: %s; falling back to live Blockscout",
            rollup_id,
            address,
            window_start_dt.isoformat(),
            window_end_exclusive_dt.isoformat(),
            exc,
        )
        return None

    row_query = (
        "SELECT\n"
        "  t.transaction_hash AS tx_hash,\n"
        "  t.block_number,\n"
        "  t.block_timestamp,\n"
        "  t.from_address,\n"
        "  t.to_address,\n"
        "  LOWER(SUBSTR(t.input, 1, 10)) AS method_id,\n"
        "  COALESCE(t.gas_price, 0) AS gas_price,\n"
        "  COALESCE(r.gas_used, 0) AS receipt_gas_used,\n"
        "  COALESCE(CAST(r.status AS STRING), '0') AS receipt_status,\n"
        "  t.value_lossless AS value_lossless,\n"
        "  t.transaction_index\n"
        f"FROM `{BIGQUERY_BLOCKSCOUT_TRANSACTIONS_TABLE}` t\n"
        f"LEFT JOIN `{BIGQUERY_BLOCKSCOUT_RECEIPTS_TABLE}` r\n"
        "  ON t.transaction_hash = r.transaction_hash\n"
        f"{where_clause}"
        "ORDER BY t.block_timestamp ASC, t.transaction_index ASC, t.transaction_hash ASC\n"
    )
    try:
        row_result = subprocess.run(
            [
                "bq",
                "query",
                f"--project_id={BIGQUERY_PROJECT_ID}",
                "--use_legacy_sql=false",
                "--format=json",
                f"--max_rows={max(expected_rows, 1)}",
            ],
            input=row_query,
            text=True,
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        logging.warning(
            "BigQuery Blockscout row query failed for %s/%s within %s..%s via project %s: %s; falling back to live Blockscout",
            rollup_id,
            address,
            window_start_dt.isoformat(),
            window_end_exclusive_dt.isoformat(),
            BIGQUERY_PROJECT_ID,
            stderr[:400] if stderr else exc,
        )
        return None
    try:
        row_payload = json.loads(row_result.stdout)
    except json.JSONDecodeError as exc:
        logging.warning(
            "BigQuery Blockscout row query returned malformed JSON for %s/%s within %s..%s: %s; falling back to live Blockscout",
            rollup_id,
            address,
            window_start_dt.isoformat(),
            window_end_exclusive_dt.isoformat(),
            exc,
        )
        return None
    if not isinstance(row_payload, list):
        logging.warning(
            "BigQuery Blockscout row query returned malformed payload for %s/%s within %s..%s: %r; falling back to live Blockscout",
            rollup_id,
            address,
            window_start_dt.isoformat(),
            window_end_exclusive_dt.isoformat(),
            row_payload,
        )
        return None

    rows: list[BlockscoutTx] = []
    for row in row_payload:
        if not isinstance(row, dict):
            logging.warning(
                "BigQuery Blockscout row query returned malformed row for %s/%s within %s..%s: %r; falling back to live Blockscout",
                rollup_id,
                address,
                window_start_dt.isoformat(),
                window_end_exclusive_dt.isoformat(),
                row,
            )
            return None
        raw_to = row.get("to_address")
        to_address = str(raw_to).strip().lower() if isinstance(raw_to, str) and raw_to.strip() else None
        raw_method_id = row.get("method_id")
        method_id = str(raw_method_id).strip().lower() if isinstance(raw_method_id, str) and raw_method_id.strip() else None
        try:
            parsed = BlockscoutTx(
                hash=str(row["tx_hash"]).strip().lower(),
                rollup_id=rollup_id,
                subtype=subtype,
                address=address,
                block_number=int(row["block_number"]),
                timestamp_utc=parse_datetime(str(row["block_timestamp"])),
                to_address=to_address,
                method_id=method_id,
                gas_price_wei=int(row["gas_price"]),
                gas_used=int(row["receipt_gas_used"]),
                value_wei=int(str(row.get("value_lossless", "0"))),
                txreceipt_status=str(row["receipt_status"]) if row.get("receipt_status") is not None else None,
            )
        except (KeyError, TypeError, ValueError, SystemExit) as exc:
            logging.warning(
                "BigQuery Blockscout row query returned malformed tx for %s/%s within %s..%s: %s; falling back to live Blockscout",
                rollup_id,
                address,
                window_start_dt.isoformat(),
                window_end_exclusive_dt.isoformat(),
                exc,
            )
            return None
        rows.append(parsed)

    if len(rows) != expected_rows:
        logging.warning(
            "BigQuery Blockscout row query returned %s rows but count query expected %s for %s/%s within %s..%s; falling back to live Blockscout",
            len(rows),
            expected_rows,
            rollup_id,
            address,
            window_start_dt.isoformat(),
            window_end_exclusive_dt.isoformat(),
        )
        return None

    page_dir = blockscout_page_dir(
        snapshot_dir=snapshot_dir,
        path_prefix=path_prefix,
        rollup_id=rollup_id,
        address=address,
        scope_id=scope_id,
    )
    window_id = blockscout_window_id(window_start_dt, window_end_exclusive_dt)
    fetched_at_utc = datetime.now(timezone.utc).isoformat()
    page_chunks = [rows[index : index + page_size] for index in range(0, len(rows), page_size)] or [[]]
    provider_labels = [BIGQUERY_BLOCKSCOUT_TRANSACTIONS_TABLE, BIGQUERY_BLOCKSCOUT_RECEIPTS_TABLE]
    query_url = (
        f"bigquery://blockchain_analytics_ethereum_mainnet_us.transactions+receipts?"
        f"filter_by={address_role}&address={address}&window_start={window_start_dt.isoformat()}&"
        f"window_end_exclusive={window_end_exclusive_dt.isoformat()}"
    )
    for page, page_rows in enumerate(page_chunks, start=1):
        page_path = page_dir / f"{window_id}_page-{page:04d}.json"
        if page_path.exists():
            logging.warning(
                "Skipping BigQuery Blockscout exact-window backfill because %s already exists; falling back to existing cache/live path",
                page_path.relative_to(repo_root()),
            )
            return None
        write_json(
            page_path,
            {
                "source": "blockscout_txlist_bigquery_backfill",
                "backfilled_via": "bigquery",
                "provider_labels": provider_labels,
                "rollup_id": rollup_id,
                "address": address,
                "filter_by": address_role,
                "method_selectors": list(method_selectors) if method_selectors else None,
                "scope_id": scope_id,
                "window_start_utc": window_start_dt.isoformat(),
                "window_end_exclusive_utc": window_end_exclusive_dt.isoformat(),
                "page": page,
                "page_size": page_size,
                "result_count": len(page_rows),
                "fetched_at_utc": fetched_at_utc,
                "url": query_url,
                "transactions": [blockscout_tx_record(tx) for tx in page_rows],
            },
        )
        request_log.append(
            {
                "source": "blockscout_txlist_bigquery_backfill",
                "rollup_id": rollup_id,
                "address": address,
                "filter_by": address_role,
                "method_selectors": list(method_selectors) if method_selectors else None,
                "scope_id": scope_id,
                "window_start_utc": window_start_dt.isoformat(),
                "window_end_exclusive_utc": window_end_exclusive_dt.isoformat(),
                "page": page,
                "page_size": page_size,
                "relative_path": str(page_path.relative_to(repo_root())),
                "fetched_at_utc": fetched_at_utc,
            }
        )

    logging.info(
        "BigQuery backfilled exact Blockscout window for %s/%s within %s..%s into %s page files (%s rows)",
        rollup_id,
        address,
        window_start_dt.isoformat(),
        window_end_exclusive_dt.isoformat(),
        len(page_chunks),
        len(rows),
    )
    return rows


def parse_blobscan_tx_record(
    row: dict[str, Any],
    *,
    rollup_id: str,
    default_subtype: str = "batchSubmissions",
) -> BlobscanTx:
    raw_to = row.get("to_address")
    to_address = str(raw_to).strip().lower() if isinstance(raw_to, str) and raw_to.strip() else None
    try:
        parsed = BlobscanTx(
            hash=str(row["hash"]).strip().lower(),
            rollup_id=str(row["rollup_id"]).strip().lower(),
            subtype=str(row.get("subtype", default_subtype)).strip(),
            block_number=int(row["block_number"]),
            timestamp_utc=parse_datetime(str(row["timestamp_utc"])),
            from_address=str(row["from_address"]).strip().lower(),
            to_address=to_address,
            blob_gas_used=int(row["blob_gas_used"]),
            blob_gas_price_wei=int(str(row["blob_gas_price_wei"])),
            blob_as_calldata_gas_used=int(row.get("blob_as_calldata_gas_used", 0)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"stored Blobscan tx record is malformed in resume path: {row!r}") from exc

    if parsed.rollup_id != rollup_id:
        raise SystemExit(f"stored Blobscan tx record does not match requested rollup {rollup_id}: {row!r}")
    if parsed.subtype not in ROLLUP_SUBTYPE_TO_COMPONENT_FIELD:
        raise SystemExit(f"stored Blobscan tx record has unknown subtype: {row!r}")
    return parsed


def load_existing_blobscan_page(
    path: Path,
    *,
    rollup_id: str,
    default_subtype: str = "batchSubmissions",
) -> tuple[list[BlobscanTx], int | None, int]:
    payload = read_json(path)
    rows = payload.get("transactions")
    if not isinstance(rows, list):
        raise SystemExit(f"stored Blobscan page is malformed: {path}")
    total_transactions = payload.get("total_transactions")
    if total_transactions is not None:
        try:
            total_transactions = int(total_transactions)
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"stored Blobscan page has malformed total_transactions: {path}") from exc
    stored_page_size = payload.get("page_size")
    try:
        stored_page_size = int(stored_page_size)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"stored Blobscan page has malformed page_size: {path}") from exc
    if stored_page_size <= 0:
        raise SystemExit(f"stored Blobscan page has non-positive page_size: {path}")
    return (
        [
            parse_blobscan_tx_record(row, rollup_id=rollup_id, default_subtype=default_subtype)
            for row in rows
            if isinstance(row, dict)
        ],
        total_transactions,
        stored_page_size,
    )


def normalize_blobscan_total_transactions(
    *,
    rollup_id: str,
    source_dir: str,
    total_transactions: int | None,
    row_count: int,
    page_size: int,
) -> int | None:
    # Blobscan count-derived totals have proven unstable on both cached and live page
    # responses. Resume safety therefore depends on exact page termination, not on any
    # stored or live total hint.
    _ = (rollup_id, source_dir, total_transactions, row_count, page_size)
    return None


def blobscan_completion_marker_path(
    *,
    snapshot_dir: Path,
    rollup_id: str,
    source_dir: str,
    window_start_dt: datetime,
    window_end_exclusive_dt: datetime,
) -> Path:
    window_id = blockscout_window_id(window_start_dt, window_end_exclusive_dt)
    return snapshot_dir / "blobscan" / rollup_id / source_dir / f"{window_id}_complete.json"


def clear_exact_blobscan_scope_cache(
    *,
    snapshot_dir: Path,
    rollup_id: str,
    source_dir: str,
    window_start_dt: datetime,
    window_end_exclusive_dt: datetime,
) -> None:
    source_path = snapshot_dir / "blobscan" / rollup_id / source_dir
    window_id = blockscout_window_id(window_start_dt, window_end_exclusive_dt)
    for page_path in source_path.glob(f"{window_id}_page-*.json"):
        page_path.unlink()
    marker_path = blobscan_completion_marker_path(
        snapshot_dir=snapshot_dir,
        rollup_id=rollup_id,
        source_dir=source_dir,
        window_start_dt=window_start_dt,
        window_end_exclusive_dt=window_end_exclusive_dt,
    )
    if marker_path.exists():
        marker_path.unlink()


def parse_blobscan_window_bounds(path: Path) -> tuple[datetime, datetime] | None:
    if path.name.endswith("_complete.json"):
        return None
    stem = path.stem
    if "_page-" not in stem:
        return None
    window_id = stem.split("_page-", 1)[0]
    start_label, end_label = window_id.split("__", 1)

    if "T" in start_label:
        start_dt = datetime.strptime(start_label, "%Y-%m-%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    else:
        start_dt = datetime_utc_start(date.fromisoformat(start_label))

    if "T" in end_label:
        end_exclusive_dt = datetime.strptime(end_label, "%Y-%m-%dT%H%M%SZ").replace(tzinfo=timezone.utc) + timedelta(
            seconds=1
        )
    else:
        end_exclusive_dt = datetime_utc_start(date.fromisoformat(end_label) + timedelta(days=1))
    return start_dt, end_exclusive_dt


def load_overlapping_blobscan_rows(
    *,
    snapshot_dir: Path,
    rollup_id: str,
    source_dir: str,
    window_start_dt: datetime,
    window_end_exclusive_dt: datetime,
) -> list[BlobscanTx]:
    source_path = snapshot_dir / "blobscan" / rollup_id / source_dir
    if not source_path.exists():
        return []

    rows_by_hash: dict[str, BlobscanTx] = {}
    for path in sorted(source_path.glob("*.json")):
        bounds = parse_blobscan_window_bounds(path)
        if bounds is None:
            continue
        cached_start_dt, cached_end_exclusive_dt = bounds
        if cached_end_exclusive_dt <= window_start_dt or cached_start_dt >= window_end_exclusive_dt:
            continue
        cached_rows, _, _ = load_existing_blobscan_page(path, rollup_id=rollup_id)
        for row in cached_rows:
            if row.hash in rows_by_hash:
                continue
            if row.timestamp_utc < window_start_dt or row.timestamp_utc >= window_end_exclusive_dt:
                continue
            rows_by_hash[row.hash] = row
    return sorted(
        rows_by_hash.values(),
        key=lambda row: (row.timestamp_utc, row.block_number, row.hash),
        reverse=True,
    )


def has_nonexact_overlapping_blobscan_pages(
    *,
    snapshot_dir: Path,
    rollup_id: str,
    source_dir: str,
    window_start_dt: datetime,
    window_end_exclusive_dt: datetime,
) -> bool:
    source_path = snapshot_dir / "blobscan" / rollup_id / source_dir
    if not source_path.exists():
        return False

    for path in source_path.glob("*.json"):
        bounds = parse_blobscan_window_bounds(path)
        if bounds is None:
            continue
        cached_start_dt, cached_end_exclusive_dt = bounds
        if cached_end_exclusive_dt <= window_start_dt or cached_start_dt >= window_end_exclusive_dt:
            continue
        if cached_start_dt == window_start_dt and cached_end_exclusive_dt == window_end_exclusive_dt:
            continue
        return True
    return False


def cached_blobscan_expected_total_transactions(
    *,
    snapshot_dir: Path,
    rollup_id: str,
    source_dir: str,
    window_start_dt: datetime,
    window_end_exclusive_dt: datetime,
) -> int | None:
    source_path = snapshot_dir / "blobscan" / rollup_id / source_dir
    if not source_path.exists():
        return None
    window_id = blockscout_window_id(window_start_dt, window_end_exclusive_dt)
    totals: list[int] = []
    for path in sorted(source_path.glob(f"{window_id}_page-*.json")):
        payload = read_json(path)
        raw_total = payload.get("total_transactions")
        if raw_total is None:
            continue
        try:
            total_transactions = int(raw_total)
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"stored Blobscan page has malformed total_transactions: {path}") from exc
        if total_transactions > 0:
            totals.append(total_transactions)
    return max(totals) if totals else None


def load_exact_blobscan_scope_cache(
    *,
    snapshot_dir: Path,
    rollup_id: str,
    source_dir: str,
    window_start_dt: datetime,
    window_end_exclusive_dt: datetime,
) -> ExactBlobscanScopeCache | None:
    source_path = snapshot_dir / "blobscan" / rollup_id / source_dir
    if not source_path.exists():
        return None

    window_id = blockscout_window_id(window_start_dt, window_end_exclusive_dt)
    page_paths = sorted(source_path.glob(f"{window_id}_page-*.json"))
    if not page_paths:
        return None

    page_numbers: list[int] = []
    page_size: int | None = None
    max_positive_total_transactions: int | None = None
    last_page_row_count = 0
    page_sources: set[str] = set()
    rows_by_hash: dict[str, BlobscanTx] = {}
    for path in page_paths:
        stem = path.stem
        if "_page-" not in stem:
            raise SystemExit(f"stored Blobscan page has malformed name: {path}")
        try:
            page_number = int(stem.rsplit("_page-", 1)[1])
        except ValueError as exc:
            raise SystemExit(f"stored Blobscan page has malformed page number: {path}") from exc
        page_numbers.append(page_number)

        payload = read_json(path)
        page_source = payload.get("source")
        if isinstance(page_source, str) and page_source:
            page_sources.add(page_source)
        page_rows, total_transactions, stored_page_size = load_existing_blobscan_page(path, rollup_id=rollup_id)
        if page_size is None:
            page_size = stored_page_size
        elif page_size != stored_page_size:
            raise SystemExit(
                f"stored Blobscan exact-scope pages have inconsistent page sizes for {rollup_id}/{source_dir} "
                f"within {window_start_dt.isoformat()}..{window_end_exclusive_dt.isoformat()}: "
                f"{page_size} vs {stored_page_size}"
            )
        if total_transactions is not None and total_transactions > 0:
            max_positive_total_transactions = max(max_positive_total_transactions or 0, total_transactions)
        last_page_row_count = len(page_rows)
        for row in page_rows:
            if row.timestamp_utc < window_start_dt or row.timestamp_utc >= window_end_exclusive_dt:
                raise SystemExit(
                    f"stored Blobscan exact-scope page contains out-of-window row for {rollup_id}/{source_dir}: {path}"
                )
            if row.hash in rows_by_hash:
                raise SystemExit(
                    f"stored Blobscan exact-scope pages contain duplicate tx hashes for {rollup_id}/{source_dir}: "
                    f"{row.hash}"
                )
            rows_by_hash[row.hash] = row

    if page_size is None:
        return None
    if len(page_sources) > 1:
        raise SystemExit(
            f"stored Blobscan exact-scope pages mix multiple sources for {rollup_id}/{source_dir} "
            f"within {window_start_dt.isoformat()}..{window_end_exclusive_dt.isoformat()}: {sorted(page_sources)}"
        )

    contiguous_pages = page_numbers == list(range(1, len(page_numbers) + 1))
    return ExactBlobscanScopeCache(
        rows=sorted(
            rows_by_hash.values(),
            key=lambda row: (row.timestamp_utc, row.block_number, row.hash),
            reverse=True,
        ),
        page_count=len(page_paths),
        page_size=page_size,
        last_page_row_count=last_page_row_count,
        max_positive_total_transactions=max_positive_total_transactions,
        contiguous_pages=contiguous_pages,
    )


def blobscan_scope_has_cached_state(
    *,
    snapshot_dir: Path,
    rollup_id: str,
    source_dir: str,
    window_start_dt: datetime,
    window_end_exclusive_dt: datetime,
) -> bool:
    marker_path = blobscan_completion_marker_path(
        snapshot_dir=snapshot_dir,
        rollup_id=rollup_id,
        source_dir=source_dir,
        window_start_dt=window_start_dt,
        window_end_exclusive_dt=window_end_exclusive_dt,
    )
    if marker_path.exists():
        return True
    source_path = snapshot_dir / "blobscan" / rollup_id / source_dir
    if not source_path.exists():
        return False
    window_id = blockscout_window_id(window_start_dt, window_end_exclusive_dt)
    if any(source_path.glob(f"{window_id}_page-*.json")):
        return True
    return has_nonexact_overlapping_blobscan_pages(
        snapshot_dir=snapshot_dir,
        rollup_id=rollup_id,
        source_dir=source_dir,
        window_start_dt=window_start_dt,
        window_end_exclusive_dt=window_end_exclusive_dt,
    )


def fetch_blobscan_total_transactions(
    *,
    rollup_id: str,
    from_address: str | None,
    rollup_filter: str | None,
    window_start_dt: datetime,
    window_end_exclusive_dt: datetime,
    retries: int,
    timeout_seconds: float,
) -> tuple[int, str, str]:
    params: dict[str, str] = {
        "startDate": iso_utc_datetime(window_start_dt),
        "endDate": iso_utc_datetime(window_end_exclusive_dt - timedelta(seconds=1)),
        "ps": "1",
        "p": "1",
        "count": "true",
    }
    if from_address:
        params["from"] = from_address
    if rollup_filter:
        params["rollups"] = blobscan_rollup_filter_value(rollup_filter)
        params["categories"] = "rollup"
    url = f"{BLOBSCAN_TX_URL}?{urlencode(params)}"
    result = fetch_json(
        url,
        headers=BLOBSCAN_HEADERS,
        retries=retries,
        timeout_seconds=timeout_seconds,
    )
    payload = result.payload
    rows = payload.get("transactions")
    if not isinstance(rows, list):
        raise SystemExit(f"Blobscan payload is malformed for {rollup_id}: {payload!r}")
    total_transactions = payload.get("totalTransactions")
    try:
        total_transactions = int(total_transactions)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"Blobscan totalTransactions is malformed for {rollup_id}: {payload!r}") from exc
    return total_transactions, result.fetched_at_utc, url


def prove_cached_blobscan_scope_complete(
    *,
    snapshot_dir: Path,
    rollup_id: str,
    source_dir: str,
    from_address: str | None,
    rollup_filter: str | None,
    window_start_dt: datetime,
    window_end_exclusive_dt: datetime,
    retries: int,
    timeout_seconds: float,
) -> list[BlobscanTx] | None:
    exact_scope_cache = load_exact_blobscan_scope_cache(
        snapshot_dir=snapshot_dir,
        rollup_id=rollup_id,
        source_dir=source_dir,
        window_start_dt=window_start_dt,
        window_end_exclusive_dt=window_end_exclusive_dt,
    )
    marker_path = blobscan_completion_marker_path(
        snapshot_dir=snapshot_dir,
        rollup_id=rollup_id,
        source_dir=source_dir,
        window_start_dt=window_start_dt,
        window_end_exclusive_dt=window_end_exclusive_dt,
    )
    if (
        exact_scope_cache is not None
        and exact_scope_cache.contiguous_pages
        and exact_scope_cache.last_page_row_count < exact_scope_cache.page_size
    ):
        write_json(
            marker_path,
            {
                "source": "blobscan_completion_marker",
                "rollup_id": rollup_id,
                "source_dir": source_dir,
                "window_start_utc": window_start_dt.isoformat(),
                "window_end_exclusive_utc": window_end_exclusive_dt.isoformat(),
                "expected_total_transactions": len(exact_scope_cache.rows),
                "cached_unique_transactions": len(exact_scope_cache.rows),
                "fetched_at_utc": None,
                "count_url": None,
                "provenance": "cached_exact_scope_terminal_page",
                "page_count": exact_scope_cache.page_count,
                "page_size": exact_scope_cache.page_size,
                "last_page_row_count": exact_scope_cache.last_page_row_count,
            },
        )
        logging.info(
            "Reusing cached Blobscan coverage for %s/%s within %s..%s after proving exact cached pages are contiguous "
            "and terminate on a short final page (%s pages, last page rows=%s, page size=%s)",
            rollup_id,
            source_dir,
            window_start_dt.isoformat(),
            window_end_exclusive_dt.isoformat(),
            exact_scope_cache.page_count,
            exact_scope_cache.last_page_row_count,
            exact_scope_cache.page_size,
        )
        return exact_scope_cache.rows

    if exact_scope_cache is not None and exact_scope_cache.contiguous_pages:
        logging.info(
            "Cached Blobscan scope for %s/%s within %s..%s still ends on a full final page (%s rows of page size %s); "
            "resuming live pagination instead of trusting count-derived totals",
            rollup_id,
            source_dir,
            window_start_dt.isoformat(),
            window_end_exclusive_dt.isoformat(),
            exact_scope_cache.last_page_row_count,
            exact_scope_cache.page_size,
        )
    return None


def maybe_reuse_completed_blobscan_scope(
    *,
    snapshot_dir: Path,
    rollup_id: str,
    source_dir: str,
    from_address: str | None,
    rollup_filter: str | None,
    window_start_dt: datetime,
    window_end_exclusive_dt: datetime,
    retries: int,
    timeout_seconds: float,
) -> list[BlobscanTx] | None:
    marker_path = blobscan_completion_marker_path(
        snapshot_dir=snapshot_dir,
        rollup_id=rollup_id,
        source_dir=source_dir,
        window_start_dt=window_start_dt,
        window_end_exclusive_dt=window_end_exclusive_dt,
    )
    if not blobscan_scope_has_cached_state(
        snapshot_dir=snapshot_dir,
        rollup_id=rollup_id,
        source_dir=source_dir,
        window_start_dt=window_start_dt,
        window_end_exclusive_dt=window_end_exclusive_dt,
    ):
        return None
    return prove_cached_blobscan_scope_complete(
        snapshot_dir=snapshot_dir,
        rollup_id=rollup_id,
        source_dir=source_dir,
        from_address=from_address,
        rollup_filter=rollup_filter,
        window_start_dt=window_start_dt,
        window_end_exclusive_dt=window_end_exclusive_dt,
        retries=retries,
        timeout_seconds=timeout_seconds,
    )


def maybe_reuse_cached_blobscan_rollup_scope_for_sender(
    *,
    snapshot_dir: Path,
    rollup_id: str,
    from_address: str | None,
    window_start_dt: datetime,
    window_end_exclusive_dt: datetime,
    retries: int,
    timeout_seconds: float,
) -> list[BlobscanTx] | None:
    if from_address is None:
        return None

    rollup_rows = maybe_reuse_completed_blobscan_scope(
        snapshot_dir=snapshot_dir,
        rollup_id=rollup_id,
        source_dir=f"rollup_{rollup_id}",
        from_address=None,
        rollup_filter=rollup_id,
        window_start_dt=window_start_dt,
        window_end_exclusive_dt=window_end_exclusive_dt,
        retries=retries,
        timeout_seconds=timeout_seconds,
    )
    if rollup_rows is None:
        return None

    filtered_rows = [row for row in rollup_rows if row.from_address == from_address]
    logging.info(
        "Reusing cached Blobscan coverage for %s/%s within %s..%s by filtering the proven-complete cached rollup scope "
        "(rollup rows=%s, sender rows=%s)",
        rollup_id,
        from_address,
        window_start_dt.isoformat(),
        window_end_exclusive_dt.isoformat(),
        len(rollup_rows),
        len(filtered_rows),
    )
    return filtered_rows


def backfill_blobscan_window_from_blockscout_receipts(
    *,
    snapshot_dir: Path,
    rollup_id: str,
    from_address: str | None,
    rollup_filter: str | None,
    window_start_dt: datetime,
    window_end_exclusive_dt: datetime,
    page_size: int,
    retries: int,
    timeout_seconds: float,
    request_log: list[dict[str, Any]],
) -> list[BlobscanTx] | None:
    source_dir = from_address or f"rollup_{rollup_filter}"
    fallback_queries: list[dict[str, Any]]
    if from_address is not None:
        fallback_queries = [
            {
                "address": from_address,
                "address_role": "from",
                "method_selectors": None,
                "path_prefix": "txlist",
                "scope_id": None,
            }
        ]
    elif rollup_filter is not None:
        registry_path = repo_root() / "registry" / "rollup_registry_v1.csv"
        registry_rollups = {row.rollup_id: row for row in load_registry(registry_path)}
        registry_rollup = registry_rollups.get(rollup_id)
        if registry_rollup is None:
            return None
        if registry_rollup.batcher_addresses:
            fallback_queries = [
                {
                    "address": address,
                    "address_role": "from",
                    "method_selectors": None,
                    "path_prefix": "txlist",
                    "scope_id": None,
                    "subtype": "batchSubmissions",
                }
                for address in registry_rollup.batcher_addresses
            ]
        elif rollup_id in ROLLUPS_WITHOUT_BATCHER_ADDRESSES:
            tracked_transactions = fetch_l2beat_tracked_transactions(
                snapshot_dir=snapshot_dir,
                rollup=registry_rollup,
                retries=retries,
                timeout_seconds=timeout_seconds,
                request_log=request_log,
            )
            tracked_calls = relevant_calldata_tracked_calls(
                rollup=registry_rollup,
                tracked_transactions=tracked_transactions,
                active_start=window_start_dt.date(),
                active_end=(window_end_exclusive_dt - timedelta(seconds=1)).date(),
            )
            if not tracked_calls:
                return None
            selectors_by_address_and_subtype: dict[tuple[str, str], set[str]] = {}
            for tracked_call in tracked_calls:
                selectors_by_address_and_subtype.setdefault(
                    (tracked_call.address, tracked_call.subtype),
                    set(),
                ).add(tracked_call.selector)
            fallback_queries = [
                {
                    "address": contract_address,
                    "address_role": "to",
                    "method_selectors": tuple(sorted(selectors)),
                    "path_prefix": "txlist_to",
                    "scope_id": "__".join(sorted(selectors)),
                    "subtype": subtype,
                }
                for (contract_address, subtype), selectors in sorted(selectors_by_address_and_subtype.items())
            ]
        else:
            return None
    else:
        return None

    logging.warning(
        "Blobscan remained unavailable for %s/%s within %s..%s; backfilling the exact scope via Blockscout txlist "
        "plus receipt enrichment across %s address hooks",
        rollup_id,
        source_dir,
        window_start_dt.isoformat(),
        window_end_exclusive_dt.isoformat(),
        len(fallback_queries),
    )

    blockscout_rows_by_hash: dict[str, BlockscoutTx] = {}
    for query in fallback_queries:
        try:
            blockscout_rows = fetch_blockscout_tx_window(
                snapshot_dir=snapshot_dir,
                rollup_id=rollup_id,
                address=query["address"],
                start_day=window_start_dt.date(),
                end_day_exclusive=window_end_exclusive_dt.date(),
                page_size=max(page_size, BLOCKSCOUT_MIN_PAGE_SIZE),
                retries=retries,
                timeout_seconds=timeout_seconds,
                request_log=request_log,
                start_timestamp_utc=window_start_dt,
                end_timestamp_exclusive_utc=window_end_exclusive_dt,
                address_role=query["address_role"],
                method_selectors=query["method_selectors"],
                path_prefix=query["path_prefix"],
                scope_id=query["scope_id"],
                subtype=query["subtype"],
            )
        except SystemExit as exc:
            logging.warning(
                "Blockscout exact-window fallback also failed for %s/%s via %s within %s..%s: %s",
                rollup_id,
                source_dir,
                query["address"],
                window_start_dt.isoformat(),
                window_end_exclusive_dt.isoformat(),
                exc,
            )
            return None
        for tx in blockscout_rows:
            blockscout_rows_by_hash.setdefault(tx.hash, tx)

    blockscout_rows = sorted(
        blockscout_rows_by_hash.values(),
        key=lambda row: (row.timestamp_utc, row.block_number, row.hash),
    )

    receipts = fetch_receipts(
        snapshot_dir=snapshot_dir,
        tx_hashes=[tx.hash for tx in blockscout_rows],
        retries=retries,
        timeout_seconds=timeout_seconds,
        batch_size=RPC_BATCH_SIZE,
        request_log=request_log,
        refresh_hashes=[tx.hash for tx in blockscout_rows],
    )

    blob_rows: list[BlobscanTx] = []
    for tx in blockscout_rows:
        receipt = receipts.get(tx.hash)
        if receipt is None:
            raise SystemExit(
                f"receipt enrichment is missing exact-window Blockscout fallback tx {tx.hash} for {rollup_id}/{source_dir}"
            )
        if receipt.blob_gas_used <= 0:
            continue
        blob_rows.append(
            BlobscanTx(
                hash=tx.hash,
                rollup_id=rollup_id,
                subtype=tx.subtype,
                block_number=tx.block_number,
                timestamp_utc=tx.timestamp_utc,
                from_address=tx.address,
                to_address=receipt.to_address or tx.to_address,
                blob_gas_used=receipt.blob_gas_used,
                blob_gas_price_wei=receipt.blob_gas_price_wei,
                # Blockscout+receipt fallback reconstructs the rent-bearing blob fields
                # but does not expose Blobscan's calldata-equivalent proxy column.
                blob_as_calldata_gas_used=0,
            )
        )

    blob_rows = sorted(
        blob_rows,
        key=lambda row: (row.timestamp_utc, row.block_number, row.hash),
        reverse=True,
    )
    page_chunks = [blob_rows[index : index + page_size] for index in range(0, len(blob_rows), page_size)] or [[]]
    window_id = blockscout_window_id(window_start_dt, window_end_exclusive_dt)
    page_dir = snapshot_dir / "blobscan" / rollup_id / source_dir
    fetched_at_utc = datetime.now(timezone.utc).isoformat()
    clear_exact_blobscan_scope_cache(
        snapshot_dir=snapshot_dir,
        rollup_id=rollup_id,
        source_dir=source_dir,
        window_start_dt=window_start_dt,
        window_end_exclusive_dt=window_end_exclusive_dt,
    )
    for page_index, page_rows in enumerate(page_chunks, start=1):
        page_path = page_dir / f"{window_id}_page-{page_index:04d}.json"
        write_json(
            page_path,
            {
                "source": "blobscan_blockscout_receipts_backfill",
                "backfilled_via": "blockscout_txlist_plus_receipts",
                "rollup_id": rollup_id,
                "from_address": from_address,
                "rollup_filter": rollup_filter,
                "window_start_utc": window_start_dt.isoformat(),
                "window_end_exclusive_utc": window_end_exclusive_dt.isoformat(),
                "page": page_index,
                "page_size": page_size,
                "total_transactions": None,
                "fetched_at_utc": fetched_at_utc,
                "transactions": [blobscan_tx_record(row) for row in page_rows],
            },
        )
        request_log.append(
            {
                "source": "blobscan_blockscout_receipts_backfill",
                "rollup_id": rollup_id,
                "from_address": from_address,
                "rollup_filter": rollup_filter,
                "window_start_utc": window_start_dt.isoformat(),
                "window_end_exclusive_utc": window_end_exclusive_dt.isoformat(),
                "page": page_index,
                "page_size": page_size,
                "relative_path": str(page_path.relative_to(repo_root())),
                "fetched_at_utc": fetched_at_utc,
            }
        )

    if blob_rows and len(blob_rows) % page_size == 0:
        terminal_page_index = len(page_chunks) + 1
        terminal_page_path = page_dir / f"{window_id}_page-{terminal_page_index:04d}.json"
        write_json(
            terminal_page_path,
            {
                "source": "blobscan_blockscout_receipts_backfill",
                "backfilled_via": "blockscout_txlist_plus_receipts",
                "rollup_id": rollup_id,
                "from_address": from_address,
                "rollup_filter": rollup_filter,
                "window_start_utc": window_start_dt.isoformat(),
                "window_end_exclusive_utc": window_end_exclusive_dt.isoformat(),
                "page": terminal_page_index,
                "page_size": page_size,
                "total_transactions": None,
                "fetched_at_utc": fetched_at_utc,
                "transactions": [],
            },
        )
        request_log.append(
            {
                "source": "blobscan_blockscout_receipts_backfill",
                "rollup_id": rollup_id,
                "from_address": from_address,
                "rollup_filter": rollup_filter,
                "window_start_utc": window_start_dt.isoformat(),
                "window_end_exclusive_utc": window_end_exclusive_dt.isoformat(),
                "page": terminal_page_index,
                "page_size": page_size,
                "relative_path": str(terminal_page_path.relative_to(repo_root())),
                "fetched_at_utc": fetched_at_utc,
            }
        )

    marker_path = blobscan_completion_marker_path(
        snapshot_dir=snapshot_dir,
        rollup_id=rollup_id,
        source_dir=source_dir,
        window_start_dt=window_start_dt,
        window_end_exclusive_dt=window_end_exclusive_dt,
    )
    write_json(
        marker_path,
        {
            "source": "blobscan_completion_marker",
            "rollup_id": rollup_id,
            "source_dir": source_dir,
            "window_start_utc": window_start_dt.isoformat(),
            "window_end_exclusive_utc": window_end_exclusive_dt.isoformat(),
            "expected_total_transactions": len(blob_rows),
            "cached_unique_transactions": len(blob_rows),
            "fetched_at_utc": fetched_at_utc,
            "count_url": None,
            "provenance": "blockscout_txlist_receipts_exact_scope",
        },
    )
    logging.info(
        "Backfilled Blobscan-exact window for %s/%s within %s..%s via Blockscout txlist plus receipts (%s blob txs)",
        rollup_id,
        source_dir,
        window_start_dt.isoformat(),
        window_end_exclusive_dt.isoformat(),
        len(blob_rows),
    )
    return blob_rows


def parse_receipt_record(row: dict[str, Any]) -> ReceiptFields:
    raw_to = row.get("to_address")
    to_address = str(raw_to).strip().lower() if isinstance(raw_to, str) and raw_to.strip() else None
    try:
        return ReceiptFields(
            hash=str(row["hash"]).strip().lower(),
            block_number=int(row["block_number"]),
            from_address=str(row["from_address"]).strip().lower(),
            to_address=to_address,
            gas_used=int(row["gas_used"]),
            effective_gas_price_wei=int(str(row["effective_gas_price_wei"])),
            blob_gas_used=int(row["blob_gas_used"]),
            blob_gas_price_wei=int(str(row["blob_gas_price_wei"])),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"stored receipt record is malformed in resume path: {row!r}") from exc


def receipt_record_payload(receipt: ReceiptFields) -> dict[str, Any]:
    return {
        "hash": receipt.hash,
        "block_number": receipt.block_number,
        "from_address": receipt.from_address,
        "to_address": receipt.to_address,
        "gas_used": receipt.gas_used,
        "effective_gas_price_wei": str(receipt.effective_gas_price_wei),
        "blob_gas_used": receipt.blob_gas_used,
        "blob_gas_price_wei": str(receipt.blob_gas_price_wei),
    }


def load_existing_receipt_batch(path: Path) -> list[ReceiptFields]:
    payload = read_json(path)
    rows = payload.get("receipts")
    if not isinstance(rows, list):
        raise SystemExit(f"stored receipt batch is malformed: {path}")
    return [parse_receipt_record(row) for row in rows if isinstance(row, dict)]


def load_existing_block_fee_batch(path: Path) -> dict[int, int]:
    payload = read_json(path)
    rows = payload.get("blocks")
    if not isinstance(rows, list):
        raise SystemExit(f"stored block fee batch is malformed: {path}")
    known: dict[int, int] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise SystemExit(f"stored block fee batch is malformed: {path}")
        try:
            known[int(row["block_number"])] = int(str(row["base_fee_per_gas_wei"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise SystemExit(f"stored block fee record is malformed in resume path: {row!r}") from exc
    return known


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def partition_checkpoint_dir(*, snapshot_dir: Path) -> Path:
    return snapshot_dir / "_runtime" / "post_partition"


def partition_checkpoint_db_path(*, snapshot_dir: Path) -> Path:
    return partition_checkpoint_dir(snapshot_dir=snapshot_dir) / "tx_universe.sqlite3"


def partition_input_watermark(*, snapshot_dir: Path) -> tuple[int, int]:
    roots = [
        snapshot_dir / "blockscout" / "txlist",
        snapshot_dir / "blockscout" / "txlist_to",
        snapshot_dir / "blobscan",
        snapshot_dir / "l2beat",
    ]
    file_count = 0
    latest_mtime_ns = 0
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            file_count += 1
            latest_mtime_ns = max(latest_mtime_ns, path.stat().st_mtime_ns)
    return file_count, latest_mtime_ns


def ensure_partition_checkpoint_schema(connection: sqlite3.Connection) -> None:
    connection.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS calldata_txs ("
        "hash TEXT PRIMARY KEY, "
        "rollup_id TEXT NOT NULL, "
        "subtype TEXT NOT NULL, "
        "block_number INTEGER NOT NULL, "
        "timestamp_utc TEXT NOT NULL, "
        "gas_price_wei TEXT NOT NULL, "
        "gas_used INTEGER NOT NULL, "
        "value_wei TEXT NOT NULL"
        ")"
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS blob_txs ("
        "hash TEXT PRIMARY KEY, "
        "rollup_id TEXT NOT NULL, "
        "subtype TEXT NOT NULL, "
        "block_number INTEGER NOT NULL, "
        "timestamp_utc TEXT NOT NULL, "
        "blob_gas_used INTEGER NOT NULL, "
        "blob_gas_price_wei TEXT NOT NULL, "
        "blob_as_calldata_gas_used INTEGER NOT NULL"
        ")"
    )
    connection.commit()


def write_partition_checkpoint(
    *,
    snapshot_dir: Path,
    registry_path: Path,
    calldata_candidate_txs_count: int,
    excluded_blob_overlap_txs: int,
    calldata_txs: dict[str, BlockscoutTx],
    blob_txs: dict[str, BlobscanTx],
    request_log: list[dict[str, Any]],
) -> Path:
    checkpoint_dir = partition_checkpoint_dir(snapshot_dir=snapshot_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = partition_checkpoint_db_path(snapshot_dir=snapshot_dir)
    tmp_path = checkpoint_path.with_name(f"{checkpoint_path.name}.tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    raw_input_file_count, raw_input_latest_mtime_ns = partition_input_watermark(snapshot_dir=snapshot_dir)
    script_path = repo_root() / "src" / "etl" / "build_l1_rent_panel.py"
    metadata_rows = [
        ("schema_version", str(PARTITION_CHECKPOINT_SCHEMA_VERSION)),
        ("checkpoint_compat_version", str(PARTITION_CHECKPOINT_COMPAT_VERSION)),
        ("created_at_utc", datetime.now(timezone.utc).isoformat()),
        ("script_sha256", sha256_file(script_path)),
        ("registry_sha256", sha256_file(registry_path)),
        ("raw_input_file_count", str(raw_input_file_count)),
        ("raw_input_latest_mtime_ns", str(raw_input_latest_mtime_ns)),
        ("calldata_candidate_txs_count", str(calldata_candidate_txs_count)),
        ("excluded_blob_overlap_txs", str(excluded_blob_overlap_txs)),
        ("calldata_txs_count", str(len(calldata_txs))),
        ("blob_txs_count", str(len(blob_txs))),
        ("request_log_json", json.dumps(request_log, sort_keys=True)),
    ]

    connection = sqlite3.connect(tmp_path)
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        ensure_partition_checkpoint_schema(connection)
        connection.executemany("INSERT INTO metadata(key, value) VALUES(?, ?)", metadata_rows)
        connection.executemany(
            "INSERT INTO calldata_txs(hash, rollup_id, subtype, block_number, timestamp_utc, gas_price_wei, gas_used, value_wei) "
            "VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
            (
                (
                    tx.hash,
                    tx.rollup_id,
                    tx.subtype,
                    tx.block_number,
                    tx.timestamp_utc.isoformat(),
                    str(tx.gas_price_wei),
                    tx.gas_used,
                    str(tx.value_wei),
                )
                for tx in calldata_txs.values()
            ),
        )
        connection.executemany(
            "INSERT INTO blob_txs(hash, rollup_id, subtype, block_number, timestamp_utc, blob_gas_used, "
            "blob_gas_price_wei, blob_as_calldata_gas_used) "
            "VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
            (
                (
                    tx.hash,
                    tx.rollup_id,
                    tx.subtype,
                    tx.block_number,
                    tx.timestamp_utc.isoformat(),
                    tx.blob_gas_used,
                    str(tx.blob_gas_price_wei),
                    tx.blob_as_calldata_gas_used,
                )
                for tx in blob_txs.values()
            ),
        )
        connection.commit()
    finally:
        connection.close()

    tmp_path.replace(checkpoint_path)
    logging.info(
        "Wrote post-partition resume checkpoint with %s calldata txs and %s blob txs to %s",
        len(calldata_txs),
        len(blob_txs),
        checkpoint_path.relative_to(repo_root()),
    )
    return checkpoint_path


def load_partition_checkpoint_if_valid(
    *,
    snapshot_dir: Path,
    registry_path: Path,
) -> PartitionCheckpoint | None:
    checkpoint_path = partition_checkpoint_db_path(snapshot_dir=snapshot_dir)
    if not checkpoint_path.exists():
        return None

    connection = sqlite3.connect(f"file:{checkpoint_path}?mode=ro", uri=True)
    try:
        connection.row_factory = sqlite3.Row
        metadata = {str(row["key"]): str(row["value"]) for row in connection.execute("SELECT key, value FROM metadata")}
        try:
            schema_version = int(metadata["schema_version"])
            checkpoint_compat_version = int(metadata.get("checkpoint_compat_version", "1"))
            raw_input_file_count = int(metadata["raw_input_file_count"])
            raw_input_latest_mtime_ns = int(metadata["raw_input_latest_mtime_ns"])
            calldata_candidate_txs_count = int(metadata["calldata_candidate_txs_count"])
            excluded_blob_overlap_txs = int(metadata["excluded_blob_overlap_txs"])
            expected_calldata_txs_count = int(metadata["calldata_txs_count"])
            expected_blob_txs_count = int(metadata["blob_txs_count"])
            request_log = json.loads(metadata["request_log_json"])
            script_sha256 = metadata["script_sha256"]
            registry_sha256 = metadata["registry_sha256"]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            logging.warning("Ignoring malformed partition checkpoint at %s: %s", checkpoint_path, exc)
            return None
        if schema_version != PARTITION_CHECKPOINT_SCHEMA_VERSION:
            logging.info(
                "Ignoring partition checkpoint at %s because schema version %s does not match %s",
                checkpoint_path,
                schema_version,
                PARTITION_CHECKPOINT_SCHEMA_VERSION,
            )
            return None
        if checkpoint_compat_version != PARTITION_CHECKPOINT_COMPAT_VERSION:
            logging.info(
                "Ignoring partition checkpoint at %s because checkpoint compatibility version %s does not match %s",
                checkpoint_path,
                checkpoint_compat_version,
                PARTITION_CHECKPOINT_COMPAT_VERSION,
            )
            return None
        current_raw_input_file_count, current_raw_input_latest_mtime_ns = partition_input_watermark(snapshot_dir=snapshot_dir)
        if (
            current_raw_input_file_count != raw_input_file_count
            or current_raw_input_latest_mtime_ns != raw_input_latest_mtime_ns
        ):
            logging.info(
                "Ignoring partition checkpoint at %s because raw tx inputs changed (cached files=%s latest_mtime_ns=%s; "
                "current files=%s latest_mtime_ns=%s)",
                checkpoint_path,
                raw_input_file_count,
                raw_input_latest_mtime_ns,
                current_raw_input_file_count,
                current_raw_input_latest_mtime_ns,
            )
            return None
        current_registry_sha256 = sha256_file(registry_path)
        if current_registry_sha256 != registry_sha256:
            logging.info(
                "Ignoring partition checkpoint at %s because registry attribution inputs changed",
                checkpoint_path,
            )
            return None
        current_script_sha256 = sha256_file(Path(__file__).resolve())
        if current_script_sha256 != script_sha256:
            logging.info(
                "Ignoring partition checkpoint at %s because the ETL script changed",
                checkpoint_path,
            )
            return None
        if not isinstance(request_log, list) or not all(isinstance(item, dict) for item in request_log):
            logging.warning("Ignoring malformed request log payload in partition checkpoint at %s", checkpoint_path)
            return None

        calldata_txs_count = int(connection.execute("SELECT COUNT(*) FROM calldata_txs").fetchone()[0])
        blob_txs_count = int(connection.execute("SELECT COUNT(*) FROM blob_txs").fetchone()[0])
        if calldata_txs_count != expected_calldata_txs_count or blob_txs_count != expected_blob_txs_count:
            logging.warning(
                "Ignoring malformed partition checkpoint at %s because row counts do not match metadata",
                checkpoint_path,
            )
            return None
    finally:
        connection.close()

    logging.info(
        "Reusing post-partition resume checkpoint from %s with %s calldata txs and %s blob txs",
        checkpoint_path.relative_to(repo_root()),
        calldata_txs_count,
        blob_txs_count,
    )
    request_log.append(
        {
            "source": "post_partition_resume_checkpoint",
            "count": calldata_txs_count + blob_txs_count,
            "relative_path": str(checkpoint_path.relative_to(repo_root())),
            "fetched_at_utc": metadata.get("created_at_utc"),
            "reused_existing": True,
        }
    )
    return PartitionCheckpoint(
        checkpoint_path=checkpoint_path,
        calldata_candidate_txs_count=calldata_candidate_txs_count,
        excluded_blob_overlap_txs=excluded_blob_overlap_txs,
        calldata_txs_count=calldata_txs_count,
        blob_txs_count=blob_txs_count,
        request_log=request_log,
    )


def receipt_lookup_db_path(*, snapshot_dir: Path) -> Path:
    return snapshot_dir / "blockscout" / "receipts_hash_lookup.sqlite3"


def block_base_fee_lookup_db_path(*, snapshot_dir: Path) -> Path:
    return snapshot_dir / "blockscout" / "block_base_fees_lookup.sqlite3"


def ensure_receipt_lookup_db_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        "CREATE TABLE IF NOT EXISTS receipts ("
        "hash TEXT PRIMARY KEY, "
        "block_number INTEGER NOT NULL, "
        "from_address TEXT NOT NULL, "
        "to_address TEXT, "
        "gas_used INTEGER NOT NULL, "
        "effective_gas_price_wei TEXT NOT NULL, "
        "blob_gas_used INTEGER NOT NULL, "
        "blob_gas_price_wei TEXT NOT NULL"
        ")"
    )
    connection.commit()


def ensure_block_base_fee_lookup_db_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        "CREATE TABLE IF NOT EXISTS block_base_fees ("
        "block_number INTEGER PRIMARY KEY, "
        "base_fee_per_gas_wei TEXT NOT NULL"
        ")"
    )
    connection.commit()


def load_receipts_from_lookup_db(*, db_path: Path, tx_hashes: Iterable[str]) -> dict[str, ReceiptFields]:
    requested_hashes = [str(tx_hash).strip().lower() for tx_hash in tx_hashes]
    if not requested_hashes:
        return {}

    known: dict[str, ReceiptFields] = {}
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        connection.row_factory = sqlite3.Row
        for hash_chunk in chunked(requested_hashes, LOOKUP_DB_SELECT_CHUNK_SIZE):
            chunk_values = [str(value) for value in hash_chunk]
            placeholders = ",".join("?" for _ in chunk_values)
            query = (
                "SELECT hash, block_number, from_address, to_address, gas_used, "
                "effective_gas_price_wei, blob_gas_used, blob_gas_price_wei "
                f"FROM receipts WHERE hash IN ({placeholders})"
            )
            for row in connection.execute(query, chunk_values):
                receipt = parse_receipt_record(
                    {
                        "hash": row["hash"],
                        "block_number": row["block_number"],
                        "from_address": row["from_address"],
                        "to_address": row["to_address"],
                        "gas_used": row["gas_used"],
                        "effective_gas_price_wei": row["effective_gas_price_wei"],
                        "blob_gas_used": row["blob_gas_used"],
                        "blob_gas_price_wei": row["blob_gas_price_wei"],
                    }
                )
                known[receipt.hash] = receipt
    finally:
        connection.close()
    return known


def load_block_base_fees_from_lookup_db(*, db_path: Path, block_numbers: Iterable[int]) -> dict[int, int]:
    requested_block_numbers = sorted({int(block_number) for block_number in block_numbers})
    if not requested_block_numbers:
        return {}

    known: dict[int, int] = {}
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        connection.row_factory = sqlite3.Row
        for block_chunk in chunked(requested_block_numbers, LOOKUP_DB_SELECT_CHUNK_SIZE):
            placeholders = ",".join("?" for _ in block_chunk)
            query = (
                "SELECT block_number, base_fee_per_gas_wei "
                f"FROM block_base_fees WHERE block_number IN ({placeholders})"
            )
            for row in connection.execute(query, [int(block_number) for block_number in block_chunk]):
                known[int(row["block_number"])] = int(str(row["base_fee_per_gas_wei"]))
    finally:
        connection.close()
    return known


def store_receipts_in_lookup_db(*, db_path: Path, receipts: Iterable[ReceiptFields]) -> int:
    rows = list(receipts)
    if not rows:
        return 0

    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    try:
        ensure_receipt_lookup_db_schema(connection)
        connection.executemany(
            "INSERT OR REPLACE INTO receipts ("
            "hash, block_number, from_address, to_address, gas_used, effective_gas_price_wei, "
            "blob_gas_used, blob_gas_price_wei"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    receipt.hash,
                    receipt.block_number,
                    receipt.from_address,
                    receipt.to_address,
                    receipt.gas_used,
                    str(receipt.effective_gas_price_wei),
                    receipt.blob_gas_used,
                    str(receipt.blob_gas_price_wei),
                )
                for receipt in rows
            ],
        )
        connection.commit()
    finally:
        connection.close()
    return len(rows)


def store_block_base_fees_in_lookup_db(*, db_path: Path, block_base_fees: Iterable[tuple[int, int]]) -> int:
    rows = [(int(block_number), int(base_fee_wei)) for block_number, base_fee_wei in block_base_fees]
    if not rows:
        return 0

    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    try:
        ensure_block_base_fee_lookup_db_schema(connection)
        connection.executemany(
            "INSERT OR REPLACE INTO block_base_fees (block_number, base_fee_per_gas_wei) VALUES (?, ?)",
            [(block_number, str(base_fee_wei)) for block_number, base_fee_wei in rows],
        )
        connection.commit()
    finally:
        connection.close()
    return len(rows)


def backfill_receipts_from_bigquery(
    *,
    snapshot_dir: Path,
    tx_hashes: Iterable[str],
    request_log: list[dict[str, Any]],
) -> dict[str, ReceiptFields]:
    requested_hashes = [str(tx_hash).strip().lower() for tx_hash in sorted(set(tx_hashes))]
    if not requested_hashes:
        return {}

    receipts: dict[str, ReceiptFields] = {}
    total_chunks = (len(requested_hashes) + BIGQUERY_RECEIPT_QUERY_CHUNK_SIZE - 1) // BIGQUERY_RECEIPT_QUERY_CHUNK_SIZE
    for chunk_index, hash_chunk in enumerate(chunked(requested_hashes, BIGQUERY_RECEIPT_QUERY_CHUNK_SIZE), start=1):
        literals = ",\n    ".join(f"'{tx_hash}'" for tx_hash in hash_chunk)
        query = (
            "SELECT\n"
            "  `hash` AS tx_hash,\n"
            "  block_number,\n"
            "  from_address,\n"
            "  to_address,\n"
            "  COALESCE(receipt_gas_used, 0) AS receipt_gas_used,\n"
            "  COALESCE(receipt_effective_gas_price, gas_price, 0) AS receipt_effective_gas_price,\n"
            "  COALESCE(receipt_blob_gas_used, 0) AS receipt_blob_gas_used,\n"
            "  COALESCE(receipt_blob_gas_price, 0) AS receipt_blob_gas_price\n"
            f"FROM `{BIGQUERY_PUBLIC_ETHEREUM_TRANSACTIONS_TABLE}`\n"
            "WHERE `hash` IN (\n"
            f"    {literals}\n"
            ")\n"
        )
        try:
            result = subprocess.run(
                [
                    "bq",
                    "query",
                    f"--project_id={BIGQUERY_PROJECT_ID}",
                    "--use_legacy_sql=false",
                    "--format=json",
                    f"--max_rows={len(hash_chunk)}",
                ],
                input=query,
                text=True,
                capture_output=True,
                check=True,
            )
        except FileNotFoundError:
            logging.warning(
                "Skipping BigQuery receipt backfill because the bq CLI is unavailable; using legacy batch cache instead"
            )
            return {}
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            logging.warning(
                "BigQuery receipt backfill failed for chunk %s/%s via project %s: %s; using legacy batch cache instead",
                chunk_index,
                total_chunks,
                BIGQUERY_PROJECT_ID,
                stderr[:400] if stderr else exc,
            )
            return {}

        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            logging.warning(
                "BigQuery receipt backfill returned malformed JSON for chunk %s/%s: %s; using legacy batch cache instead",
                chunk_index,
                total_chunks,
                exc,
            )
            return {}
        if not isinstance(payload, list):
            logging.warning(
                "BigQuery receipt backfill returned malformed payload for chunk %s/%s: %r; using legacy batch cache instead",
                chunk_index,
                total_chunks,
                payload,
            )
            return {}

        for row in payload:
            if not isinstance(row, dict):
                logging.warning(
                    "BigQuery receipt backfill returned malformed row for chunk %s/%s: %r; using legacy batch cache instead",
                    chunk_index,
                    total_chunks,
                    row,
                )
                return {}
            try:
                receipt = parse_receipt_record(
                    {
                        "hash": row["tx_hash"],
                        "block_number": row["block_number"],
                        "from_address": row["from_address"],
                        "to_address": row.get("to_address"),
                        "gas_used": row["receipt_gas_used"],
                        "effective_gas_price_wei": row["receipt_effective_gas_price"],
                        "blob_gas_used": row["receipt_blob_gas_used"],
                        "blob_gas_price_wei": row["receipt_blob_gas_price"],
                    }
                )
            except SystemExit as exc:
                logging.warning(
                    "BigQuery receipt backfill returned malformed receipt row for chunk %s/%s: %s; using legacy batch cache instead",
                    chunk_index,
                    total_chunks,
                    exc,
                )
                return {}
            receipts[receipt.hash] = receipt

        if (
            chunk_index == 1
            or chunk_index == total_chunks
            or chunk_index % BIGQUERY_RECEIPT_PROGRESS_LOG_INTERVAL == 0
        ):
            logging.info(
                "BigQuery receipt backfill progress %s/%s chunks (%s/%s requested tx hashes, %s rows returned so far)",
                chunk_index,
                total_chunks,
                min(chunk_index * BIGQUERY_RECEIPT_QUERY_CHUNK_SIZE, len(requested_hashes)),
                len(requested_hashes),
                len(receipts),
            )

    rows_inserted = store_receipts_in_lookup_db(
        db_path=receipt_lookup_db_path(snapshot_dir=snapshot_dir),
        receipts=receipts.values(),
    )
    fetched_at_utc = datetime.now(timezone.utc).isoformat()
    provenance_path = (
        snapshot_dir
        / "blockscout"
        / f"receipts_hash_lookup_bigquery_backfill_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    write_json(
        provenance_path,
        {
            "source": "bigquery_public_crypto_ethereum_transactions_receipts_backfill",
            "project_id": BIGQUERY_PROJECT_ID,
            "table": BIGQUERY_PUBLIC_ETHEREUM_TRANSACTIONS_TABLE,
            "fetched_at_utc": fetched_at_utc,
            "requested_hashes": len(requested_hashes),
            "rows_returned": len(receipts),
            "rows_inserted": rows_inserted,
            "missing_after_bigquery": len([tx_hash for tx_hash in requested_hashes if tx_hash not in receipts]),
        },
    )
    request_log.append(
        {
            "source": "bigquery_public_crypto_ethereum.transactions",
            "count": len(requested_hashes),
            "rows_returned": len(receipts),
            "relative_path": str(provenance_path.relative_to(repo_root())),
            "fetched_at_utc": fetched_at_utc,
        }
    )
    logging.info(
        "BigQuery backfilled %s/%s requested receipt rows into %s",
        len(receipts),
        len(requested_hashes),
        receipt_lookup_db_path(snapshot_dir=snapshot_dir).relative_to(repo_root()),
    )
    return receipts


def backfill_block_base_fees_from_bigquery(
    *,
    snapshot_dir: Path,
    block_numbers: Iterable[int],
    request_log: list[dict[str, Any]],
) -> dict[int, int]:
    requested_block_numbers = sorted({int(block_number) for block_number in block_numbers})
    if not requested_block_numbers:
        return {}

    known: dict[int, int] = {}
    attempted_chunks = 0
    successful_chunks = 0
    requested_block_number_set = set(requested_block_numbers)
    min_block_number = requested_block_numbers[0]
    max_block_number = requested_block_numbers[-1]
    block_span = max_block_number - min_block_number + 1
    density = Decimal(len(requested_block_numbers)) / Decimal(block_span)
    chunk_specs: list[tuple[str, int, int, list[int] | None, int]] = []
    if len(requested_block_numbers) > BIGQUERY_BLOCK_BASE_FEE_QUERY_CHUNK_SIZE:
        requested_index = 0
        while requested_index < len(requested_block_numbers):
            chunk_start = requested_block_numbers[requested_index]
            range_end = min(chunk_start + BIGQUERY_BLOCK_BASE_FEE_RANGE_SIZE - 1, max_block_number)
            lookahead_index = requested_index
            while lookahead_index < len(requested_block_numbers) and requested_block_numbers[lookahead_index] <= range_end:
                lookahead_index += 1
            chunk_requested_count = lookahead_index - requested_index
            local_density = Decimal(chunk_requested_count) / Decimal(range_end - chunk_start + 1)
            if (
                chunk_requested_count > BIGQUERY_BLOCK_BASE_FEE_QUERY_CHUNK_SIZE
                and local_density >= Decimal(str(BIGQUERY_BLOCK_BASE_FEE_RANGE_DENSITY_THRESHOLD))
            ):
                chunk_specs.append(("range", chunk_start, range_end, None, chunk_requested_count))
                requested_index = lookahead_index
                continue
            block_chunk = requested_block_numbers[
                requested_index : min(requested_index + BIGQUERY_BLOCK_BASE_FEE_QUERY_CHUNK_SIZE, len(requested_block_numbers))
            ]
            chunk_specs.append(("point", block_chunk[0], block_chunk[-1], block_chunk, len(block_chunk)))
            requested_index += len(block_chunk)
        range_chunk_count = sum(1 for chunk_mode, *_ in chunk_specs if chunk_mode == "range")
        point_chunk_count = len(chunk_specs) - range_chunk_count
        logging.info(
            "BigQuery block-base-fee backfill selected hybrid mode for %s requested blocks across %s..%s (global density %.3f, %s range chunks, %s point chunks)",
            len(requested_block_numbers),
            min_block_number,
            max_block_number,
            float(density),
            range_chunk_count,
            point_chunk_count,
        )
    else:
        for block_chunk in chunked(requested_block_numbers, BIGQUERY_BLOCK_BASE_FEE_QUERY_CHUNK_SIZE):
            chunk_specs.append(("point", block_chunk[0], block_chunk[-1], [int(value) for value in block_chunk], len(block_chunk)))
        logging.info(
            "BigQuery block-base-fee backfill selected point mode for %s requested blocks across %s..%s (density %.3f, %s chunks of size <= %s)",
            len(requested_block_numbers),
            min_block_number,
            max_block_number,
            float(density),
            len(chunk_specs),
            BIGQUERY_BLOCK_BASE_FEE_QUERY_CHUNK_SIZE,
        )

    total_chunks = len(chunk_specs)
    for chunk_index, (chunk_mode, chunk_start, chunk_end, block_chunk, chunk_requested_count) in enumerate(chunk_specs, start=1):
        attempted_chunks += 1
        if chunk_mode == "range":
            query = (
                "SELECT number AS block_number, base_fee_per_gas\n"
                f"FROM `{BIGQUERY_PUBLIC_ETHEREUM_BLOCKS_TABLE}`\n"
                f"WHERE number BETWEEN {chunk_start} AND {chunk_end}\n"
                "ORDER BY number ASC\n"
            )
            max_rows = chunk_end - chunk_start + 1
        else:
            if block_chunk is None:
                raise SystemExit("point-mode block-base-fee backfill chunk is missing its requested block list")
            query = (
                "SELECT number AS block_number, base_fee_per_gas\n"
                f"FROM `{BIGQUERY_PUBLIC_ETHEREUM_BLOCKS_TABLE}`\n"
                f"WHERE number IN UNNEST([{', '.join(str(block_number) for block_number in block_chunk)}])\n"
                "ORDER BY number ASC\n"
            )
            max_rows = len(block_chunk)
        try:
            result = subprocess.run(
                [
                    "bq",
                    "query",
                    f"--project_id={BIGQUERY_PROJECT_ID}",
                    "--use_legacy_sql=false",
                    "--format=json",
                    f"--max_rows={max_rows}",
                ],
                input=query,
                text=True,
                capture_output=True,
                check=True,
            )
        except FileNotFoundError:
            logging.warning(
                "Skipping BigQuery block-base-fee backfill because the bq CLI is unavailable; falling back to cached/live RPC",
            )
            return known
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            logging.warning(
                "BigQuery block-base-fee %s query failed for %s requested blocks via project %s: %s; falling back to cached/live RPC for this chunk",
                chunk_mode,
                chunk_requested_count,
                BIGQUERY_PROJECT_ID,
                stderr[:400] if stderr else exc,
            )
            continue
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            logging.warning(
                "BigQuery block-base-fee %s query returned malformed JSON for %s requested blocks: %s; falling back to cached/live RPC for this chunk",
                chunk_mode,
                chunk_requested_count,
                exc,
            )
            continue
        if not isinstance(payload, list):
            logging.warning(
                "BigQuery block-base-fee %s query returned malformed payload for %s requested blocks: %r; falling back to cached/live RPC for this chunk",
                chunk_mode,
                chunk_requested_count,
                payload,
            )
            continue

        chunk_rows: dict[int, int] = {}
        malformed_chunk = False
        for row in payload:
            if not isinstance(row, dict):
                malformed_chunk = True
                break
            try:
                block_number = int(row["block_number"])
                base_fee_wei = int(str(row["base_fee_per_gas"]))
            except (KeyError, TypeError, ValueError):
                malformed_chunk = True
                break
            if block_number not in requested_block_number_set:
                continue
            chunk_rows[block_number] = base_fee_wei
        if malformed_chunk:
            logging.warning(
                "BigQuery block-base-fee %s query returned malformed rows for %s requested blocks; falling back to cached/live RPC for this chunk",
                chunk_mode,
                chunk_requested_count,
            )
            continue

        known.update(chunk_rows)
        successful_chunks += 1
        if chunk_index == 1 or chunk_index == total_chunks or chunk_index % 10 == 0:
            logging.info(
                "BigQuery block-base-fee backfill progress %s/%s chunks (%s/%s requested blocks resolved so far)",
                chunk_index,
                total_chunks,
                len(known),
                len(requested_block_numbers),
            )

    if not known:
        return {}

    rows_inserted = store_block_base_fees_in_lookup_db(
        db_path=block_base_fee_lookup_db_path(snapshot_dir=snapshot_dir),
        block_base_fees=known.items(),
    )
    fetched_at_utc = datetime.now(timezone.utc).isoformat()
    provenance_path = (
        snapshot_dir
        / "blockscout"
        / f"block_base_fees_bigquery_backfill_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    write_json(
        provenance_path,
        {
            "source": "bigquery_public_crypto_ethereum_blocks_base_fee_backfill",
            "project_id": BIGQUERY_PROJECT_ID,
            "table": BIGQUERY_PUBLIC_ETHEREUM_BLOCKS_TABLE,
            "fetched_at_utc": fetched_at_utc,
            "requested_blocks": len(requested_block_numbers),
            "rows_returned": len(known),
            "rows_inserted": rows_inserted,
            "missing_after_bigquery": len(
                [block_number for block_number in requested_block_numbers if block_number not in known]
            ),
            "mode": "hybrid" if len(requested_block_numbers) > BIGQUERY_BLOCK_BASE_FEE_QUERY_CHUNK_SIZE else "point",
            "range_size": BIGQUERY_BLOCK_BASE_FEE_RANGE_SIZE if len(requested_block_numbers) > BIGQUERY_BLOCK_BASE_FEE_QUERY_CHUNK_SIZE else None,
            "density": format(density, "f"),
            "attempted_chunks": attempted_chunks,
            "successful_chunks": successful_chunks,
        },
    )
    request_log.append(
        {
            "source": "bigquery_public_crypto_ethereum.blocks",
            "count": len(requested_block_numbers),
            "rows_returned": len(known),
            "relative_path": str(provenance_path.relative_to(repo_root())),
            "fetched_at_utc": fetched_at_utc,
        }
    )
    logging.info(
        "BigQuery backfilled %s/%s requested block base fees into %s",
        len(known),
        len(requested_block_numbers),
        block_base_fee_lookup_db_path(snapshot_dir=snapshot_dir).relative_to(repo_root()),
    )
    return known


def fetch_json(
    url: str,
    *,
    headers: dict[str, str],
    retries: int,
    timeout_seconds: float,
    method: str = "GET",
    body: bytes | None = None,
) -> FetchResult:
    delay_seconds = 1.0
    for attempt in range(1, retries + 1):
        request = Request(url, data=body, headers=headers, method=method)
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                raw_bytes = response.read()
            fetched_at = datetime.now(timezone.utc).isoformat()
            payload = json.loads(raw_bytes.decode("utf-8"))
            return FetchResult(payload=payload, raw_bytes=raw_bytes, fetched_at_utc=fetched_at)
        except HTTPError as exc:
            body_text = exc.read(200).decode("utf-8", errors="replace")
            logging.warning("%s attempt %s/%s failed with HTTP %s: %r", url, attempt, retries, exc.code, body_text)
        except (
            URLError,
            TimeoutError,
            json.JSONDecodeError,
            http.client.IncompleteRead,
            http.client.RemoteDisconnected,
            ConnectionResetError,
        ) as exc:
            logging.warning("%s attempt %s/%s failed: %s", url, attempt, retries, exc)

        if attempt < retries:
            time.sleep(delay_seconds)
            delay_seconds *= 2.0

    raise SystemExit(f"source instability or breaking API changes while fetching {url}")


def fetch_text(
    url: str,
    *,
    headers: dict[str, str],
    retries: int,
    timeout_seconds: float,
    method: str = "GET",
    body: bytes | None = None,
) -> FetchResult:
    delay_seconds = 1.0
    for attempt in range(1, retries + 1):
        request = Request(url, data=body, headers=headers, method=method)
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                raw_bytes = response.read()
            fetched_at = datetime.now(timezone.utc).isoformat()
            payload = raw_bytes.decode("utf-8")
            return FetchResult(payload=payload, raw_bytes=raw_bytes, fetched_at_utc=fetched_at)
        except HTTPError as exc:
            body_text = exc.read(200).decode("utf-8", errors="replace")
            logging.warning("%s attempt %s/%s failed with HTTP %s: %r", url, attempt, retries, exc.code, body_text)
        except (
            URLError,
            TimeoutError,
            UnicodeDecodeError,
            http.client.IncompleteRead,
            http.client.RemoteDisconnected,
            ConnectionResetError,
        ) as exc:
            logging.warning("%s attempt %s/%s failed: %s", url, attempt, retries, exc)

        if attempt < retries:
            time.sleep(delay_seconds)
            delay_seconds *= 2.0

    raise SystemExit(f"source instability or breaking API changes while fetching {url}")


def fetch_eth_rpc_json(
    *,
    body: bytes,
    retries: int,
    timeout_seconds: float,
    provider_url: str,
) -> FetchResult:
    return fetch_json(
        provider_url,
        headers={"Content-Type": "application/json", **BLOCKSCOUT_HEADERS},
        retries=retries,
        timeout_seconds=timeout_seconds,
        method="POST",
        body=body,
    )


def parse_hex_quantity(value: Any, *, field_name: str) -> int:
    if value is None:
        raise ValueError(f"{field_name} is null")
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} is empty")
    return int(text, 16)


def parse_rpc_receipt(receipt: dict[str, Any]) -> ReceiptFields:
    raw_to = receipt.get("to")
    to_address = str(raw_to).strip().lower() if isinstance(raw_to, str) and raw_to.strip() else None
    return ReceiptFields(
        hash=str(receipt["transactionHash"]).strip().lower(),
        block_number=parse_hex_quantity(receipt["blockNumber"], field_name="blockNumber"),
        from_address=str(receipt["from"]).strip().lower(),
        to_address=to_address,
        gas_used=parse_hex_quantity(receipt["gasUsed"], field_name="gasUsed"),
        effective_gas_price_wei=parse_hex_quantity(receipt["effectiveGasPrice"], field_name="effectiveGasPrice"),
        blob_gas_used=parse_hex_quantity(receipt.get("blobGasUsed", "0x0"), field_name="blobGasUsed"),
        blob_gas_price_wei=parse_hex_quantity(receipt.get("blobGasPrice", "0x0"), field_name="blobGasPrice"),
    )


def parse_rpc_block_record(block: dict[str, Any]) -> tuple[int, int, str]:
    block_number = parse_hex_quantity(block["number"], field_name="number")
    base_fee_wei = parse_hex_quantity(block.get("baseFeePerGas", "0x0"), field_name="baseFeePerGas")
    timestamp_utc = datetime.fromtimestamp(
        parse_hex_quantity(block["timestamp"], field_name="timestamp"),
        tz=timezone.utc,
    ).isoformat()
    return block_number, base_fee_wei, timestamp_utc


def timestamp_to_utc_date(value: int) -> date:
    return datetime.fromtimestamp(value, tz=timezone.utc).date()


def tracked_function_call_record(row: TrackedFunctionCall) -> dict[str, Any]:
    return {
        "rollup_id": row.rollup_id,
        "subtype": row.subtype,
        "address": row.address,
        "selector": row.selector,
        "signature": row.signature,
        "since_timestamp": row.since_timestamp,
        "until_timestamp": row.until_timestamp,
    }


def parse_l2beat_tracked_call(row: dict[str, Any], *, rollup_id: str, subtype: str) -> TrackedFunctionCall | None:
    params = row.get("params")
    if not isinstance(params, dict):
        return None
    if str(params.get("formula", "")).strip() != "functionCall":
        return None

    raw_address = params.get("address")
    raw_selector = params.get("selector")
    if not isinstance(raw_address, str) or not raw_address.strip():
        return None
    if not isinstance(raw_selector, str) or not raw_selector.strip():
        return None

    raw_since = row.get("sinceTimestamp")
    if raw_since is None:
        return None
    try:
        since_timestamp = int(raw_since)
        until_timestamp = int(row["untilTimestamp"]) if row.get("untilTimestamp") is not None else None
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"L2BEAT tracked transaction timestamps are malformed for {rollup_id}: {row!r}") from exc

    return TrackedFunctionCall(
        rollup_id=rollup_id,
        subtype=subtype,
        address=raw_address.strip().lower(),
        selector=raw_selector.strip().lower(),
        signature=str(params.get("signature", "")).strip(),
        since_timestamp=since_timestamp,
        until_timestamp=until_timestamp,
    )


def extract_l2beat_tracked_transactions(html: str, *, evidence_url: str, rollup_id: str) -> dict[str, list[TrackedFunctionCall]]:
    marker = "window.__SSR_DATA__="
    marker_index = html.find(marker)
    if marker_index < 0:
        raise SystemExit(f"L2BEAT page is missing SSR data for {rollup_id}: {evidence_url}")
    json_start = marker_index + len(marker)
    json_end = html.find("</script>", json_start)
    if json_end < 0:
        raise SystemExit(f"L2BEAT page is missing closing SSR script for {rollup_id}: {evidence_url}")

    try:
        data = json.loads(html[json_start:json_end])
    except json.JSONDecodeError as exc:
        raise SystemExit(f"L2BEAT SSR payload is malformed for {rollup_id}: {evidence_url}") from exc

    sections = data.get("props", {}).get("projectEntry", {}).get("sections")
    if not isinstance(sections, list):
        raise SystemExit(f"L2BEAT project sections are missing for {rollup_id}: {evidence_url}")

    tracked_transactions: dict[str, Any] | None = None
    for section in sections:
        if not isinstance(section, dict):
            continue
        props = section.get("props")
        if not isinstance(props, dict):
            continue
        candidate = props.get("trackedTransactions")
        if isinstance(candidate, dict):
            tracked_transactions = candidate
            break
    if tracked_transactions is None:
        raise SystemExit(f"L2BEAT tracked transactions are missing for {rollup_id}: {evidence_url}")

    parsed: dict[str, list[TrackedFunctionCall]] = {}
    for subtype in ("batchSubmissions", "proofSubmissions", "stateUpdates"):
        rows = tracked_transactions.get(subtype, [])
        if not isinstance(rows, list):
            raise SystemExit(f"L2BEAT tracked transaction subtype is malformed for {rollup_id}: {subtype}")
        parsed_rows = [
            tracked
            for tracked in (
                parse_l2beat_tracked_call(row, rollup_id=rollup_id, subtype=subtype)
                for row in rows
                if isinstance(row, dict)
            )
            if tracked is not None
        ]
        parsed[subtype] = parsed_rows
    return parsed


def load_existing_l2beat_tracked_transactions(path: Path, *, rollup_id: str, evidence_url: str) -> dict[str, list[TrackedFunctionCall]]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise SystemExit(f"stored L2BEAT tracked transaction snapshot is malformed: {path}")
    if payload.get("rollup_id") != rollup_id:
        raise SystemExit(f"stored L2BEAT tracked transaction snapshot has wrong rollup_id: {path}")
    if payload.get("evidence_url") != evidence_url:
        raise SystemExit(f"stored L2BEAT tracked transaction snapshot has wrong evidence_url: {path}")

    raw_tracked = payload.get("tracked_transactions")
    if not isinstance(raw_tracked, dict):
        raise SystemExit(f"stored L2BEAT tracked transaction snapshot is malformed: {path}")

    parsed: dict[str, list[TrackedFunctionCall]] = {}
    for subtype in TRACKED_TRANSACTION_SUBTYPES:
        rows = raw_tracked.get(subtype, [])
        if not isinstance(rows, list):
            raise SystemExit(f"stored L2BEAT tracked transaction subtype is malformed: {path}")
        parsed_rows: list[TrackedFunctionCall] = []
        for row in rows:
            if not isinstance(row, dict):
                raise SystemExit(f"stored L2BEAT tracked transaction snapshot is malformed: {path}")
            try:
                parsed_rows.append(
                    TrackedFunctionCall(
                        rollup_id=str(row["rollup_id"]).strip(),
                        subtype=str(row["subtype"]).strip(),
                        address=str(row["address"]).strip().lower(),
                        selector=str(row["selector"]).strip().lower(),
                        signature=str(row.get("signature", "")).strip(),
                        since_timestamp=int(row["since_timestamp"]),
                        until_timestamp=int(row["until_timestamp"]) if row.get("until_timestamp") is not None else None,
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise SystemExit(f"stored L2BEAT tracked transaction snapshot is malformed: {path}") from exc
        parsed[subtype] = parsed_rows
    return parsed


def supplement_tracked_transactions(
    *,
    rollup_id: str,
    tracked_transactions: dict[str, list[TrackedFunctionCall]],
) -> dict[str, list[TrackedFunctionCall]]:
    supplemented = {
        subtype: list(tracked_transactions.get(subtype, []))
        for subtype in TRACKED_TRANSACTION_SUBTYPES
    }
    for tracked_call in LEGACY_TRACKED_CALLS_BY_ROLLUP.get(rollup_id, ()):
        rows = supplemented.setdefault(tracked_call.subtype, [])
        if tracked_call not in rows:
            rows.append(tracked_call)
    for subtype in TRACKED_TRANSACTION_SUBTYPES:
        supplemented[subtype] = sorted(
            supplemented.get(subtype, []),
            key=lambda row: (
                row.address,
                row.selector,
                row.since_timestamp,
                row.until_timestamp if row.until_timestamp is not None else 2**63 - 1,
            ),
        )
    return supplemented


def tracked_call_pairs(
    tracked_transactions: dict[str, list[TrackedFunctionCall]],
) -> list[tuple[str, str]]:
    return sorted(
        {
            (row.address.lower(), row.selector.lower())
            for subtype in TRACKED_TRANSACTION_SUBTYPES
            for row in tracked_transactions.get(subtype, [])
        }
    )


def load_existing_bigquery_tracked_call_observations(
    path: Path,
    *,
    rollup_id: str,
    observed_end: date,
    expected_pairs: list[tuple[str, str]],
) -> dict[tuple[str, str], tuple[int, int, int]]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise SystemExit(f"stored tracked-call observation cache is malformed: {path}")
    if payload.get("rollup_id") != rollup_id:
        raise SystemExit(f"stored tracked-call observation cache has wrong rollup_id: {path}")
    if payload.get("observed_end_utc") != observed_end.isoformat():
        raise SystemExit(f"stored tracked-call observation cache has wrong observed_end_utc: {path}")

    raw_requested_pairs = payload.get("requested_pairs")
    if not isinstance(raw_requested_pairs, list):
        raise SystemExit(f"stored tracked-call observation cache is malformed: {path}")
    cached_pairs: list[tuple[str, str]] = []
    for row in raw_requested_pairs:
        if not isinstance(row, dict):
            raise SystemExit(f"stored tracked-call observation cache is malformed: {path}")
        address = str(row.get("address", "")).strip().lower()
        selector = str(row.get("selector", "")).strip().lower()
        if not address or not selector:
            raise SystemExit(f"stored tracked-call observation cache has malformed requested_pairs: {path}")
        cached_pairs.append((address, selector))
    if sorted(cached_pairs) != expected_pairs:
        raise SystemExit(
            f"stored tracked-call observation cache does not match the current selector set for {rollup_id}: {path}"
        )

    raw_observed_pairs = payload.get("observed_pairs")
    if not isinstance(raw_observed_pairs, list):
        raise SystemExit(f"stored tracked-call observation cache is malformed: {path}")

    observed_pairs: dict[tuple[str, str], tuple[int, int, int]] = {}
    for row in raw_observed_pairs:
        if not isinstance(row, dict):
            raise SystemExit(f"stored tracked-call observation cache is malformed: {path}")
        try:
            address = str(row["address"]).strip().lower()
            selector = str(row["selector"]).strip().lower()
            first_seen_timestamp = int(row["first_seen_timestamp"])
            last_seen_timestamp = int(row["last_seen_timestamp"])
            tx_count = int(row["tx_count"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SystemExit(f"stored tracked-call observation cache is malformed: {path}") from exc
        observed_pairs[(address, selector)] = (first_seen_timestamp, last_seen_timestamp, tx_count)
    return observed_pairs


def fetch_bigquery_tracked_call_observations(
    *,
    snapshot_dir: Path,
    rollup_id: str,
    tracked_transactions: dict[str, list[TrackedFunctionCall]],
    observed_end: date,
    request_log: list[dict[str, Any]],
) -> dict[tuple[str, str], tuple[int, int, int]] | None:
    requested_pairs = tracked_call_pairs(tracked_transactions)
    if not requested_pairs:
        return {}

    path = snapshot_dir / "bigquery" / "tracked_call_observations" / rollup_id / "observed_windows.json"
    if path.exists():
        try:
            observed_pairs = load_existing_bigquery_tracked_call_observations(
                path,
                rollup_id=rollup_id,
                observed_end=observed_end,
                expected_pairs=requested_pairs,
            )
        except SystemExit as exc:
            logging.warning(
                "Discarding stale tracked-call observation cache for %s because it no longer matches the "
                "current selector universe: %s",
                rollup_id,
                exc,
            )
        else:
            request_log.append(
                {
                    "source": "bigquery_tracked_call_observations",
                    "rollup_id": rollup_id,
                    "relative_path": str(path.relative_to(repo_root())),
                    "observed_end_utc": observed_end.isoformat(),
                    "fetched_at_utc": None,
                    "reused_existing": True,
                }
            )
            return observed_pairs

    selectors_by_address: dict[str, set[str]] = {}
    for address, selector in requested_pairs:
        selectors_by_address.setdefault(address, set()).add(selector)

    clauses: list[str] = []
    for address, selectors in sorted(selectors_by_address.items()):
        selector_literals = ", ".join(f"'{selector}'" for selector in sorted(selectors))
        clauses.append(
            f"(LOWER(to_address) = '{address}' AND LOWER(SUBSTR(input, 1, 10)) IN ({selector_literals}))"
        )

    observed_end_exclusive_dt = datetime_utc_start(observed_end + timedelta(days=1))
    query = (
        "SELECT\n"
        "  LOWER(to_address) AS address,\n"
        "  LOWER(SUBSTR(input, 1, 10)) AS selector,\n"
        "  MIN(UNIX_SECONDS(block_timestamp)) AS first_seen_timestamp,\n"
        "  MAX(UNIX_SECONDS(block_timestamp)) AS last_seen_timestamp,\n"
        "  COUNT(*) AS tx_count\n"
        f"FROM `{BIGQUERY_PUBLIC_ETHEREUM_TRANSACTIONS_TABLE}`\n"
        f"WHERE block_timestamp < TIMESTAMP('{observed_end_exclusive_dt.isoformat()}')\n"
        "  AND (\n    "
        + "\n    OR ".join(clauses)
        + "\n  )\n"
        "GROUP BY address, selector\n"
        "ORDER BY address, selector\n"
    )

    try:
        result = subprocess.run(
            [
                "bq",
                "query",
                f"--project_id={BIGQUERY_PROJECT_ID}",
                "--use_legacy_sql=false",
                "--format=json",
                f"--max_rows={max(len(requested_pairs), 1)}",
            ],
            input=query,
            text=True,
            capture_output=True,
            check=True,
        )
    except FileNotFoundError:
        logging.warning(
            "Skipping BigQuery tracked-call observation query because the bq CLI is unavailable; leaving %s tracked-call windows unbounded",
            rollup_id,
        )
        return None
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        logging.warning(
            "BigQuery tracked-call observation query failed for %s via project %s: %s; leaving tracked-call windows unbounded",
            rollup_id,
            BIGQUERY_PROJECT_ID,
            stderr[:400] if stderr else exc,
        )
        return None

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        logging.warning(
            "BigQuery tracked-call observation query returned malformed JSON for %s: %s; leaving tracked-call windows unbounded",
            rollup_id,
            exc,
        )
        return None
    if not isinstance(payload, list):
        logging.warning(
            "BigQuery tracked-call observation query returned malformed payload for %s: %r; leaving tracked-call windows unbounded",
            rollup_id,
            payload,
        )
        return None

    observed_pairs: dict[tuple[str, str], tuple[int, int, int]] = {}
    observed_rows: list[dict[str, Any]] = []
    for row in payload:
        if not isinstance(row, dict):
            logging.warning(
                "BigQuery tracked-call observation query returned malformed row for %s: %r; leaving tracked-call windows unbounded",
                rollup_id,
                row,
            )
            return None
        try:
            address = str(row["address"]).strip().lower()
            selector = str(row["selector"]).strip().lower()
            first_seen_timestamp = int(row["first_seen_timestamp"])
            last_seen_timestamp = int(row["last_seen_timestamp"])
            tx_count = int(row["tx_count"])
        except (KeyError, TypeError, ValueError) as exc:
            logging.warning(
                "BigQuery tracked-call observation query returned malformed values for %s: %s; leaving tracked-call windows unbounded",
                rollup_id,
                exc,
            )
            return None
        observed_pairs[(address, selector)] = (first_seen_timestamp, last_seen_timestamp, tx_count)
        observed_rows.append(
            {
                "address": address,
                "selector": selector,
                "first_seen_timestamp": first_seen_timestamp,
                "last_seen_timestamp": last_seen_timestamp,
                "tx_count": tx_count,
            }
        )

    fetched_at_utc = datetime.now(timezone.utc).isoformat()
    write_json(
        path,
        {
            "source": "bigquery_tracked_call_observations",
            "rollup_id": rollup_id,
            "observed_end_utc": observed_end.isoformat(),
            "table": BIGQUERY_PUBLIC_ETHEREUM_TRANSACTIONS_TABLE,
            "fetched_at_utc": fetched_at_utc,
            "requested_pairs": [
                {"address": address, "selector": selector}
                for address, selector in requested_pairs
            ],
            "observed_pairs": observed_rows,
        },
    )
    request_log.append(
        {
            "source": "bigquery_tracked_call_observations",
            "rollup_id": rollup_id,
            "relative_path": str(path.relative_to(repo_root())),
            "observed_end_utc": observed_end.isoformat(),
            "fetched_at_utc": fetched_at_utc,
        }
    )
    return observed_pairs


def bound_tracked_transactions_to_observed_history(
    *,
    snapshot_dir: Path,
    rollup_id: str,
    tracked_transactions: dict[str, list[TrackedFunctionCall]],
    observed_end: date,
    request_log: list[dict[str, Any]],
) -> dict[str, list[TrackedFunctionCall]]:
    observed_pairs = fetch_bigquery_tracked_call_observations(
        snapshot_dir=snapshot_dir,
        rollup_id=rollup_id,
        tracked_transactions=tracked_transactions,
        observed_end=observed_end,
        request_log=request_log,
    )
    if observed_pairs is None:
        return tracked_transactions

    bounded: dict[str, list[TrackedFunctionCall]] = {subtype: [] for subtype in TRACKED_TRANSACTION_SUBTYPES}
    dropped_descriptions: list[str] = []
    clamped_count = 0
    for subtype in TRACKED_TRANSACTION_SUBTYPES:
        for row in tracked_transactions.get(subtype, []):
            observed = observed_pairs.get((row.address, row.selector))
            if observed is None:
                dropped_descriptions.append(f"{subtype}:{row.address}/{row.selector}")
                continue
            first_seen_timestamp, last_seen_timestamp, _tx_count = observed
            bounded_since_timestamp = max(row.since_timestamp, first_seen_timestamp)
            bounded_until_timestamp = last_seen_timestamp
            if row.until_timestamp is not None:
                bounded_until_timestamp = min(bounded_until_timestamp, row.until_timestamp)
            if bounded_until_timestamp < bounded_since_timestamp:
                dropped_descriptions.append(f"{subtype}:{row.address}/{row.selector}")
                continue
            if (
                bounded_since_timestamp != row.since_timestamp
                or bounded_until_timestamp != row.until_timestamp
            ):
                clamped_count += 1
            bounded[subtype].append(
                TrackedFunctionCall(
                    rollup_id=row.rollup_id,
                    subtype=row.subtype,
                    address=row.address,
                    selector=row.selector,
                    signature=row.signature,
                    since_timestamp=bounded_since_timestamp,
                    until_timestamp=bounded_until_timestamp,
                )
            )

    for subtype in TRACKED_TRANSACTION_SUBTYPES:
        bounded[subtype] = sorted(
            bounded.get(subtype, []),
            key=lambda row: (
                row.address,
                row.selector,
                row.since_timestamp,
                row.until_timestamp if row.until_timestamp is not None else 2**63 - 1,
            ),
        )

    if clamped_count:
        logging.info(
            "Clamped %s tracked-call windows for %s to observed on-chain first/last use through %s",
            clamped_count,
            rollup_id,
            observed_end.isoformat(),
        )
    if dropped_descriptions:
        logging.info(
            "Dropped %s tracked-call selectors with no observed on-chain usage for %s through %s: %s",
            len(dropped_descriptions),
            rollup_id,
            observed_end.isoformat(),
            ", ".join(dropped_descriptions[:12]),
        )
    return bounded


def fetch_l2beat_tracked_transactions(
    *,
    snapshot_dir: Path,
    rollup: RegistryRollup,
    observed_end: date,
    retries: int,
    timeout_seconds: float,
    request_log: list[dict[str, Any]],
) -> dict[str, list[TrackedFunctionCall]]:
    if not rollup.evidence_url:
        raise SystemExit(f"required registry evidence_url is missing for {rollup.rollup_id}")
    path = snapshot_dir / "l2beat" / rollup.rollup_id / "tracked_transactions.json"
    if path.exists():
        tracked_transactions = supplement_tracked_transactions(
            rollup_id=rollup.rollup_id,
            tracked_transactions=load_existing_l2beat_tracked_transactions(
            path,
            rollup_id=rollup.rollup_id,
            evidence_url=rollup.evidence_url,
            ),
        )
        request_log.append(
            {
                "source": "l2beat_project_page_tracked_transactions",
                "rollup_id": rollup.rollup_id,
                "evidence_url": rollup.evidence_url,
                "relative_path": str(path.relative_to(repo_root())),
                "fetched_at_utc": None,
                "reused_existing": True,
            }
        )
        return bound_tracked_transactions_to_observed_history(
            snapshot_dir=snapshot_dir,
            rollup_id=rollup.rollup_id,
            tracked_transactions=tracked_transactions,
            observed_end=observed_end,
            request_log=request_log,
        )

    result = fetch_text(
        rollup.evidence_url,
        headers=BROWSER_HEADERS,
        retries=retries,
        timeout_seconds=timeout_seconds,
    )
    tracked_transactions = extract_l2beat_tracked_transactions(
        result.payload,
        evidence_url=rollup.evidence_url,
        rollup_id=rollup.rollup_id,
    )
    write_json(
        path,
        {
            "source": "l2beat_project_page_tracked_transactions",
            "rollup_id": rollup.rollup_id,
            "evidence_url": rollup.evidence_url,
            "fetched_at_utc": result.fetched_at_utc,
            "page_sha256": hashlib.sha256(result.raw_bytes).hexdigest(),
            "tracked_transactions": {
                subtype: [tracked_function_call_record(row) for row in rows]
                for subtype, rows in tracked_transactions.items()
            },
        },
    )
    request_log.append(
        {
            "source": "l2beat_project_page_tracked_transactions",
            "rollup_id": rollup.rollup_id,
            "evidence_url": rollup.evidence_url,
            "relative_path": str(path.relative_to(repo_root())),
            "fetched_at_utc": result.fetched_at_utc,
        }
    )
    return bound_tracked_transactions_to_observed_history(
        snapshot_dir=snapshot_dir,
        rollup_id=rollup.rollup_id,
        tracked_transactions=supplement_tracked_transactions(
            rollup_id=rollup.rollup_id,
            tracked_transactions=tracked_transactions,
        ),
        observed_end=observed_end,
        request_log=request_log,
    )


def relevant_calldata_tracked_calls(
    *,
    rollup: RegistryRollup,
    tracked_transactions: dict[str, list[TrackedFunctionCall]],
    active_start: date,
    active_end: date,
) -> list[TrackedFunctionCall]:
    if active_end < active_start:
        return []

    relevant: list[TrackedFunctionCall] = []
    for subtype in ("batchSubmissions", "proofSubmissions", "stateUpdates"):
        for row in tracked_transactions.get(subtype, []):
            since_date = timestamp_to_utc_date(row.since_timestamp)
            until_date = timestamp_to_utc_date(row.until_timestamp) if row.until_timestamp is not None else None
            if since_date > active_end:
                continue
            if until_date is not None and until_date < active_start:
                continue
            relevant.append(row)
    return relevant


def relevant_pre_dencun_tracked_calls(
    *,
    rollup: RegistryRollup,
    tracked_transactions: dict[str, list[TrackedFunctionCall]],
    active_start: date,
    pre_dencun_end: date,
) -> list[TrackedFunctionCall]:
    return relevant_calldata_tracked_calls(
        rollup=rollup,
        tracked_transactions=tracked_transactions,
        active_start=active_start,
        active_end=pre_dencun_end,
    )


def load_registry(path: Path) -> list[RegistryRollup]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    rollups: list[RegistryRollup] = []
    for row in rows:
        raw_addresses = row.get("batcher_addresses_json", "")
        try:
            parsed_addresses = json.loads(raw_addresses) if raw_addresses else []
        except json.JSONDecodeError as exc:
            raise SystemExit(f"invalid batcher_addresses_json for {row.get('rollup_id')}: {raw_addresses!r}") from exc
        if not isinstance(parsed_addresses, list):
            raise SystemExit(f"batcher_addresses_json must be a JSON list for {row.get('rollup_id')}")

        addresses = tuple(
            str(address).strip().lower()
            for address in parsed_addresses
            if isinstance(address, str) and str(address).strip()
        )
        rollups.append(
            RegistryRollup(
                rollup_id=row["rollup_id"],
                start_date_utc=parse_date(row["start_date_utc"]),
                end_date_utc=parse_date(row["end_date_utc"]) if row["end_date_utc"] else None,
                batcher_addresses=addresses,
                evidence_url=str(row.get("evidence_url", "")).strip(),
            )
        )
    return rollups


def observed_end_date(run_date: date) -> date:
    return run_date - timedelta(days=1)


def month_windows(start_day: date, end_day: date) -> list[tuple[date, date]]:
    if end_day < start_day:
        return []

    windows: list[tuple[date, date]] = []
    cursor = date(start_day.year, start_day.month, 1)
    while cursor <= end_day:
        if cursor.month == 12:
            next_month = date(cursor.year + 1, 1, 1)
        else:
            next_month = date(cursor.year, cursor.month + 1, 1)
        window_start = max(start_day, cursor)
        window_end_exclusive = min(end_day + timedelta(days=1), next_month)
        windows.append((window_start, window_end_exclusive))
        cursor = next_month
    return windows


def window_label(start_day: date, end_day_exclusive: date) -> str:
    return f"{start_day.isoformat()}__{(end_day_exclusive - timedelta(days=1)).isoformat()}"


def iso_utc_start(day: date) -> str:
    return datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")


def iso_utc_end_inclusive(end_day_exclusive: date) -> str:
    end_dt = datetime.combine(end_day_exclusive, datetime.min.time(), tzinfo=timezone.utc) - timedelta(seconds=1)
    return end_dt.isoformat().replace("+00:00", "Z")


def datetime_utc_start(day: date) -> datetime:
    return datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc)


def iso_utc_datetime(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def blockscout_window_id(start_dt: datetime, end_dt_exclusive: datetime) -> str:
    if start_dt.timetz() == datetime.min.time().replace(tzinfo=timezone.utc) and end_dt_exclusive.timetz() == datetime.min.time().replace(tzinfo=timezone.utc):
        return window_label(start_dt.date(), end_dt_exclusive.date())

    end_inclusive = end_dt_exclusive - timedelta(seconds=1)
    return (
        f"{start_dt.strftime('%Y-%m-%dT%H%M%SZ')}__"
        f"{end_inclusive.strftime('%Y-%m-%dT%H%M%SZ')}"
    )


def normalize_blockscout_tx(
    row: dict[str, Any],
    *,
    rollup_id: str,
    address: str,
    subtype: str = "batchSubmissions",
    address_role: str = "from",
    method_selectors: tuple[str, ...] | None = None,
) -> BlockscoutTx | None:
    matched_address = str(row.get(address_role, "")).strip().lower()
    if matched_address != address:
        return None

    raw_to = row.get("to")
    to_address = str(raw_to).strip().lower() if isinstance(raw_to, str) and raw_to.strip() else None
    raw_method_id = row.get("methodId")
    if isinstance(raw_method_id, str) and raw_method_id.strip():
        method_id = raw_method_id.strip().lower()
    else:
        raw_input = row.get("input")
        if isinstance(raw_input, str) and raw_input.startswith("0x") and len(raw_input) >= 10:
            method_id = raw_input[:10].lower()
        else:
            method_id = None
    if method_selectors is not None and method_id not in method_selectors:
        return None

    try:
        return BlockscoutTx(
            hash=str(row["hash"]).strip().lower(),
            rollup_id=rollup_id,
            subtype=subtype,
            address=address,
            block_number=int(str(row["blockNumber"])),
            timestamp_utc=datetime.fromtimestamp(int(str(row["timeStamp"])), tz=timezone.utc),
            to_address=to_address,
            method_id=method_id,
            gas_price_wei=int(str(row["gasPrice"])),
            gas_used=int(str(row["gasUsed"])),
            value_wei=int(str(row.get("value", "0"))),
            txreceipt_status=str(row["txreceipt_status"]) if row.get("txreceipt_status") is not None else None,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"Blockscout txlist row is malformed for {rollup_id}/{address}: {row!r}") from exc


def blockscout_tx_record(tx: BlockscoutTx) -> dict[str, Any]:
    return {
        "hash": tx.hash,
        "rollup_id": tx.rollup_id,
        "subtype": tx.subtype,
        "address": tx.address,
        "block_number": tx.block_number,
        "timestamp_utc": tx.timestamp_utc.isoformat(),
        "to_address": tx.to_address,
        "method_id": tx.method_id,
        "gas_price_wei": str(tx.gas_price_wei),
        "gas_used": tx.gas_used,
        "value_wei": str(tx.value_wei),
        "txreceipt_status": tx.txreceipt_status,
    }


def fetch_blockscout_tx_window(
    *,
    snapshot_dir: Path,
    rollup_id: str,
    address: str,
    start_day: date,
    end_day_exclusive: date,
    page_size: int,
    retries: int,
    timeout_seconds: float,
    request_log: list[dict[str, Any]],
    start_timestamp_utc: datetime | None = None,
    end_timestamp_exclusive_utc: datetime | None = None,
    address_role: str = "from",
    method_selectors: tuple[str, ...] | None = None,
    path_prefix: str = "txlist",
    scope_id: str | None = None,
    subtype: str = "batchSubmissions",
) -> list[BlockscoutTx]:
    window_start_dt = start_timestamp_utc or datetime_utc_start(start_day)
    window_end_exclusive_dt = end_timestamp_exclusive_utc or datetime_utc_start(end_day_exclusive)
    if window_end_exclusive_dt <= window_start_dt:
        return []

    window_id = blockscout_window_id(window_start_dt, window_end_exclusive_dt)
    max_pages = max(1, 10000 // page_size)
    page = 1
    all_rows: list[BlockscoutTx] = []
    cached_page_size: int | None = None
    window_span_seconds = int((window_end_exclusive_dt - window_start_dt).total_seconds())

    while True:
        page_dir = blockscout_page_dir(
            snapshot_dir=snapshot_dir,
            path_prefix=path_prefix,
            rollup_id=rollup_id,
            address=address,
            scope_id=scope_id,
        )
        page_path = page_dir / f"{window_id}_page-{page:04d}.json"
        if page_path.exists():
            compact_rows, stored_page_size, stored_result_count = load_existing_blockscout_page(
                page_path,
                rollup_id=rollup_id,
                address=address,
                default_subtype=subtype,
            )
            cached_page_size = stored_page_size
            request_log.append(
                {
                    "source": "blockscout_txlist",
                    "rollup_id": rollup_id,
                    "address": address,
                    "filter_by": address_role,
                    "method_selectors": list(method_selectors) if method_selectors else None,
                    "scope_id": scope_id,
                    "window_start_utc": window_start_dt.isoformat(),
                    "window_end_exclusive_utc": window_end_exclusive_dt.isoformat(),
                    "page": page,
                    "page_size": page_size,
                    "relative_path": str(page_path.relative_to(repo_root())),
                    "fetched_at_utc": None,
                    "reused_existing": True,
                }
            )
            all_rows.extend(compact_rows)
            if stored_result_count < stored_page_size:
                break
            page += 1
            continue

        has_overlapping_cache = page == 1 and overlapping_blockscout_cache_exists(
            page_dir=page_dir,
            window_start_dt=window_start_dt,
            window_end_exclusive_dt=window_end_exclusive_dt,
            window_id=window_id,
        )
        if has_overlapping_cache:
            reused_rows = maybe_reuse_blockscout_window_from_overlapping_cache(
                page_dir=page_dir,
                rollup_id=rollup_id,
                address=address,
                window_start_dt=window_start_dt,
                window_end_exclusive_dt=window_end_exclusive_dt,
                window_id=window_id,
                request_log=request_log,
                address_role=address_role,
                method_selectors=method_selectors,
                scope_id=scope_id,
                subtype=subtype,
            )
            if reused_rows is not None:
                return reused_rows
            logging.info(
                "Attempting exact-window BigQuery backfill for %s/%s within %s..%s because overlapping Blockscout cache exists but %s is missing",
                rollup_id,
                address,
                window_start_dt.isoformat(),
                window_end_exclusive_dt.isoformat(),
                page_path.name,
            )
            bigquery_rows = backfill_blockscout_window_from_bigquery(
                snapshot_dir=snapshot_dir,
                rollup_id=rollup_id,
                address=address,
                window_start_dt=window_start_dt,
                window_end_exclusive_dt=window_end_exclusive_dt,
                page_size=page_size,
                request_log=request_log,
                address_role=address_role,
                method_selectors=method_selectors,
                path_prefix=path_prefix,
                scope_id=scope_id,
                subtype=subtype,
            )
            if bigquery_rows is not None:
                return bigquery_rows
        elif page == 1:
            logging.info(
                "Attempting exact-window BigQuery backfill for %s/%s within %s..%s because %s is uncached",
                rollup_id,
                address,
                window_start_dt.isoformat(),
                window_end_exclusive_dt.isoformat(),
                page_path.name,
            )
            bigquery_rows = backfill_blockscout_window_from_bigquery(
                snapshot_dir=snapshot_dir,
                rollup_id=rollup_id,
                address=address,
                window_start_dt=window_start_dt,
                window_end_exclusive_dt=window_end_exclusive_dt,
                page_size=page_size,
                request_log=request_log,
                address_role=address_role,
                method_selectors=method_selectors,
                path_prefix=path_prefix,
                scope_id=scope_id,
                subtype=subtype,
            )
            if bigquery_rows is not None:
                return bigquery_rows

        if page > 1 and cached_page_size is not None and cached_page_size != page_size:
            if not all_rows:
                raise SystemExit(
                    f"cannot resume cached Blockscout window with a different page size before any rows were loaded "
                    f"for {rollup_id}/{address}"
                )
            continuation_start_dt = all_rows[-1].timestamp_utc
            resume_page_size = cached_page_size
            logging.info(
                "Resuming Blockscout tx window for %s/%s from %s with cached page size %s after requested page size %s "
                "within %s..%s",
                rollup_id,
                address,
                continuation_start_dt.isoformat(),
                resume_page_size,
                page_size,
                window_start_dt.isoformat(),
                window_end_exclusive_dt.isoformat(),
            )
            continuation_rows = fetch_blockscout_tx_window(
                snapshot_dir=snapshot_dir,
                rollup_id=rollup_id,
                address=address,
                start_day=continuation_start_dt.date(),
                end_day_exclusive=window_end_exclusive_dt.date(),
                page_size=resume_page_size,
                retries=retries,
                timeout_seconds=timeout_seconds,
                request_log=request_log,
                start_timestamp_utc=continuation_start_dt,
                end_timestamp_exclusive_utc=window_end_exclusive_dt,
                address_role=address_role,
                method_selectors=method_selectors,
                path_prefix=path_prefix,
                scope_id=scope_id,
                subtype=subtype,
            )
            seen_hashes = {row.hash for row in all_rows}
            for row in continuation_rows:
                if row.hash in seen_hashes:
                    continue
                seen_hashes.add(row.hash)
                all_rows.append(row)
            return all_rows

        if page > max_pages:
            if not all_rows:
                raise SystemExit(
                    f"Blockscout txlist result window exceeded {max_pages * page_size} rows "
                    f"before any rows were captured for {rollup_id}/{address}"
                )
            continuation_start_dt = all_rows[-1].timestamp_utc
            if continuation_start_dt <= window_start_dt and window_span_seconds <= 1:
                raise SystemExit(
                    f"Blockscout txlist result window exceeded {max_pages * page_size} rows within a one-second span "
                    f"for {rollup_id}/{address}"
                )
            logging.info(
                "Continuing dense Blockscout tx window for %s/%s from %s within %s..%s after %s rows",
                rollup_id,
                address,
                continuation_start_dt.isoformat(),
                window_start_dt.isoformat(),
                window_end_exclusive_dt.isoformat(),
                len(all_rows),
            )
            continuation_rows = fetch_blockscout_tx_window(
                snapshot_dir=snapshot_dir,
                rollup_id=rollup_id,
                address=address,
                start_day=continuation_start_dt.date(),
                end_day_exclusive=window_end_exclusive_dt.date(),
                page_size=page_size,
                retries=retries,
                timeout_seconds=timeout_seconds,
                request_log=request_log,
                start_timestamp_utc=continuation_start_dt,
                end_timestamp_exclusive_utc=window_end_exclusive_dt,
                address_role=address_role,
                method_selectors=method_selectors,
                path_prefix=path_prefix,
                scope_id=scope_id,
                subtype=subtype,
            )
            seen_hashes = {row.hash for row in all_rows}
            for row in continuation_rows:
                if row.hash in seen_hashes:
                    continue
                seen_hashes.add(row.hash)
                all_rows.append(row)
            return all_rows

        params = {
            "module": "account",
            "action": "txlist",
            "address": address,
            "filter_by": address_role,
            "start_timestamp": str(int(window_start_dt.timestamp())),
            "end_timestamp": str(int((window_end_exclusive_dt - timedelta(seconds=1)).timestamp())),
            "page": str(page),
            "offset": str(page_size),
            "sort": "asc",
        }
        url = f"{BLOCKSCOUT_TXLIST_URL}?{urlencode(params)}"
        try:
            result = fetch_json(
                url,
                headers=BLOCKSCOUT_HEADERS,
                retries=retries,
                timeout_seconds=timeout_seconds,
            )
        except SystemExit:
            split_start_dt = all_rows[-1].timestamp_utc if all_rows else window_start_dt
            fallback_page_size = max(BLOCKSCOUT_MIN_PAGE_SIZE, page_size // 2)
            if fallback_page_size < page_size:
                logging.info(
                    "Retrying Blockscout tx window for %s/%s with smaller page size %s after page %s failure at page size %s within %s..%s",
                    rollup_id,
                    address,
                    fallback_page_size,
                    page,
                    page_size,
                    split_start_dt.isoformat(),
                    window_end_exclusive_dt.isoformat(),
                )
                fallback_rows = fetch_blockscout_tx_window(
                    snapshot_dir=snapshot_dir,
                    rollup_id=rollup_id,
                    address=address,
                    start_day=split_start_dt.date(),
                    end_day_exclusive=window_end_exclusive_dt.date(),
                    page_size=fallback_page_size,
                    retries=retries,
                    timeout_seconds=timeout_seconds,
                    request_log=request_log,
                    start_timestamp_utc=split_start_dt,
                    end_timestamp_exclusive_utc=window_end_exclusive_dt,
                    address_role=address_role,
                    method_selectors=method_selectors,
                    path_prefix=path_prefix,
                    scope_id=scope_id,
                    subtype=subtype,
                )
                merged_rows = list(all_rows)
                seen_hashes = {row.hash for row in merged_rows}
                for row in fallback_rows:
                    if row.hash in seen_hashes:
                        continue
                    seen_hashes.add(row.hash)
                    merged_rows.append(row)
                return merged_rows
            remaining_span_seconds = int((window_end_exclusive_dt - split_start_dt).total_seconds())
            if remaining_span_seconds > 1:
                split_offset_seconds = max(1, remaining_span_seconds // 2)
                split_dt = split_start_dt + timedelta(seconds=split_offset_seconds)
                logging.info(
                    "Splitting slow Blockscout tx window for %s/%s from %s..%s after page %s failure into %s..%s and %s..%s",
                    rollup_id,
                    address,
                    window_start_dt.isoformat(),
                    window_end_exclusive_dt.isoformat(),
                    page,
                    split_start_dt.isoformat(),
                    split_dt.isoformat(),
                    split_dt.isoformat(),
                    window_end_exclusive_dt.isoformat(),
                )
                merged_rows = list(all_rows)
                left_rows = fetch_blockscout_tx_window(
                    snapshot_dir=snapshot_dir,
                    rollup_id=rollup_id,
                    address=address,
                    start_day=split_start_dt.date(),
                    end_day_exclusive=split_dt.date(),
                    page_size=page_size,
                    retries=retries,
                    timeout_seconds=timeout_seconds,
                    request_log=request_log,
                    start_timestamp_utc=split_start_dt,
                    end_timestamp_exclusive_utc=split_dt,
                    address_role=address_role,
                    method_selectors=method_selectors,
                    path_prefix=path_prefix,
                    scope_id=scope_id,
                    subtype=subtype,
                )
                right_rows = fetch_blockscout_tx_window(
                    snapshot_dir=snapshot_dir,
                    rollup_id=rollup_id,
                    address=address,
                    start_day=split_dt.date(),
                    end_day_exclusive=window_end_exclusive_dt.date(),
                    page_size=page_size,
                    retries=retries,
                    timeout_seconds=timeout_seconds,
                    request_log=request_log,
                    start_timestamp_utc=split_dt,
                    end_timestamp_exclusive_utc=window_end_exclusive_dt,
                    address_role=address_role,
                    method_selectors=method_selectors,
                    path_prefix=path_prefix,
                    scope_id=scope_id,
                    subtype=subtype,
                )
                seen_hashes = {row.hash for row in merged_rows}
                for row in left_rows + right_rows:
                    if row.hash in seen_hashes:
                        continue
                    seen_hashes.add(row.hash)
                    merged_rows.append(row)
                return merged_rows
            raise
        payload = result.payload
        rows = payload.get("result")
        if not isinstance(rows, list):
            raise SystemExit(f"Blockscout txlist payload is malformed for {rollup_id}/{address}: {payload!r}")

        normalized_rows = [
            normalize_blockscout_tx(
                row,
                rollup_id=rollup_id,
                address=address,
                subtype=subtype,
                address_role=address_role,
                method_selectors=method_selectors,
            )
            for row in rows
            if isinstance(row, dict)
        ]
        compact_rows = [tx for tx in normalized_rows if tx is not None]
        write_json(
            page_path,
            {
                "source": "blockscout_txlist",
                "rollup_id": rollup_id,
                "address": address,
                "filter_by": address_role,
                "method_selectors": list(method_selectors) if method_selectors else None,
                "scope_id": scope_id,
                "window_start_utc": window_start_dt.isoformat(),
                "window_end_exclusive_utc": window_end_exclusive_dt.isoformat(),
                "page": page,
                "page_size": page_size,
                "result_count": len(rows),
                "fetched_at_utc": result.fetched_at_utc,
                "url": url,
                "transactions": [blockscout_tx_record(tx) for tx in compact_rows],
            },
        )
        request_log.append(
            {
                "source": "blockscout_txlist",
                "rollup_id": rollup_id,
                "address": address,
                "filter_by": address_role,
                "method_selectors": list(method_selectors) if method_selectors else None,
                "scope_id": scope_id,
                "window_start_utc": window_start_dt.isoformat(),
                "window_end_exclusive_utc": window_end_exclusive_dt.isoformat(),
                "page": page,
                "page_size": page_size,
                "relative_path": str(page_path.relative_to(repo_root())),
                "fetched_at_utc": result.fetched_at_utc,
            }
        )
        all_rows.extend(compact_rows)
        if len(rows) < page_size:
            break
        page += 1

    return all_rows


def normalize_blobscan_tx(
    tx: dict[str, Any],
    *,
    default_rollup_id: str | None = None,
    subtype: str = "batchSubmissions",
) -> BlobscanTx:
    raw_rollup = tx.get("rollup")
    rollup_id = (
        canonicalize_blobscan_rollup_id(str(raw_rollup))
        if isinstance(raw_rollup, str) and raw_rollup.strip()
        else normalize_slug(default_rollup_id or "")
    )
    if not rollup_id:
        raise SystemExit(f"Blobscan transaction is missing rollup attribution: {tx!r}")

    raw_to = tx.get("to")
    to_address = str(raw_to).strip().lower() if isinstance(raw_to, str) and raw_to.strip() else None

    try:
        return BlobscanTx(
            hash=str(tx["hash"]).strip().lower(),
            rollup_id=rollup_id,
            subtype=subtype,
            block_number=int(tx["blockNumber"]),
            timestamp_utc=parse_datetime(str(tx["blockTimestamp"])),
            from_address=str(tx["from"]).strip().lower(),
            to_address=to_address,
            blob_gas_used=int(str(tx["blobGasUsed"])),
            blob_gas_price_wei=int(str(tx["blobGasPrice"])),
            blob_as_calldata_gas_used=int(str(tx.get("blobAsCalldataGasUsed", "0"))),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"Blobscan transaction row is malformed: {tx!r}") from exc


def blobscan_tx_record(tx: BlobscanTx) -> dict[str, Any]:
    return {
        "hash": tx.hash,
        "rollup_id": tx.rollup_id,
        "subtype": tx.subtype,
        "block_number": tx.block_number,
        "timestamp_utc": tx.timestamp_utc.isoformat(),
        "from_address": tx.from_address,
        "to_address": tx.to_address,
        "blob_gas_used": tx.blob_gas_used,
        "blob_gas_price_wei": str(tx.blob_gas_price_wei),
        "blob_as_calldata_gas_used": tx.blob_as_calldata_gas_used,
    }


def canonicalize_blobscan_rollup_id(value: str) -> str:
    normalized = normalize_slug(value)
    return BLOBSCAN_ROLLUP_ALIASES.get(normalized, normalized)


def blobscan_rollup_filter_value(rollup_id: str) -> str:
    normalized = normalize_slug(rollup_id)
    for alias, canonical in BLOBSCAN_ROLLUP_ALIASES.items():
        if canonical == normalized:
            return alias
    return normalized


def fetch_blobscan_window(
    *,
    snapshot_dir: Path,
    rollup_id: str,
    start_day: date,
    end_day_exclusive: date,
    page_size: int,
    retries: int,
    timeout_seconds: float,
    request_log: list[dict[str, Any]],
    from_address: str | None = None,
    rollup_filter: str | None = None,
    start_timestamp_utc: datetime | None = None,
    end_timestamp_exclusive_utc: datetime | None = None,
    instability_retries_remaining: int = BLOBSCAN_INSTABILITY_RETRY_ROUNDS,
    terminal_window_retries_remaining: int = BLOBSCAN_TERMINAL_WINDOW_RETRY_ROUNDS,
    allow_rollup_discovery_fallback: bool = True,
    subtype: str = "batchSubmissions",
) -> list[BlobscanTx]:
    if not from_address and not rollup_filter:
        raise ValueError("blobscan fetch requires from_address or rollup_filter")

    window_start_dt = start_timestamp_utc or datetime_utc_start(start_day)
    window_end_exclusive_dt = end_timestamp_exclusive_utc or datetime_utc_start(end_day_exclusive)
    if window_end_exclusive_dt <= window_start_dt:
        return []

    window_end_inclusive_dt = window_end_exclusive_dt - timedelta(seconds=1)
    page = 1
    total_rows: list[BlobscanTx] = []
    total_transactions: int | None = None
    cached_page_size: int | None = None
    window_id = blockscout_window_id(window_start_dt, window_end_exclusive_dt)
    source_dir = from_address or f"rollup_{rollup_filter}"

    completed_rows = maybe_reuse_completed_blobscan_scope(
        snapshot_dir=snapshot_dir,
        rollup_id=rollup_id,
        source_dir=source_dir,
        from_address=from_address,
        rollup_filter=rollup_filter,
        window_start_dt=window_start_dt,
        window_end_exclusive_dt=window_end_exclusive_dt,
        retries=retries,
        timeout_seconds=timeout_seconds,
    )
    if completed_rows is not None:
        return completed_rows
    if from_address is not None and rollup_filter is None:
        rollup_scope_rows = maybe_reuse_cached_blobscan_rollup_scope_for_sender(
            snapshot_dir=snapshot_dir,
            rollup_id=rollup_id,
            from_address=from_address,
            window_start_dt=window_start_dt,
            window_end_exclusive_dt=window_end_exclusive_dt,
            retries=retries,
            timeout_seconds=timeout_seconds,
        )
        if rollup_scope_rows is not None:
            request_log.append(
                {
                    "source": "blobscan_rollup_scope_filter",
                    "rollup_id": rollup_id,
                    "from_address": from_address,
                    "window_start_utc": window_start_dt.isoformat(),
                    "window_end_exclusive_utc": window_end_exclusive_dt.isoformat(),
                    "reused_existing": True,
                }
            )
            return rollup_scope_rows

    while True:
        page_path = snapshot_dir / "blobscan" / rollup_id / source_dir / f"{window_id}_page-{page:04d}.json"
        if page_path.exists():
            normalized_rows, existing_total, stored_page_size = load_existing_blobscan_page(
                page_path,
                rollup_id=rollup_id,
                default_subtype=subtype,
            )
            existing_total = normalize_blobscan_total_transactions(
                rollup_id=rollup_id,
                source_dir=source_dir,
                total_transactions=existing_total,
                row_count=len(normalized_rows),
                page_size=stored_page_size,
            )
            if cached_page_size is None:
                cached_page_size = stored_page_size
            elif cached_page_size != stored_page_size:
                raise SystemExit(
                    f"stored Blobscan pages within one resume scope have inconsistent page sizes for {rollup_id}/{source_dir}: "
                    f"{cached_page_size} vs {stored_page_size}"
                )
            if total_transactions is None:
                total_transactions = existing_total
            if (
                page == 1
                and not total_rows
                and from_address is not None
                and rollup_filter is None
                and allow_rollup_discovery_fallback
                and existing_total is not None
                and existing_total > 0
                and not normalized_rows
            ):
                logging.warning(
                    "Stored Blobscan page for %s/%s is empty despite total_transactions=%s; retrying the same "
                    "scope via rollups=%s instead of from=%s",
                    rollup_id,
                    source_dir,
                    existing_total,
                    rollup_id,
                    from_address,
                )
                return fetch_blobscan_window(
                    snapshot_dir=snapshot_dir,
                    rollup_id=rollup_id,
                    start_day=window_start_dt.date(),
                    end_day_exclusive=window_end_exclusive_dt.date(),
                    page_size=page_size,
                    retries=retries,
                    timeout_seconds=timeout_seconds,
                    request_log=request_log,
                    rollup_filter=rollup_id,
                    start_timestamp_utc=window_start_dt,
                    end_timestamp_exclusive_utc=window_end_exclusive_dt,
                    allow_rollup_discovery_fallback=allow_rollup_discovery_fallback,
                    subtype=subtype,
                )
            request_log.append(
                {
                    "source": "blobscan_transactions",
                    "rollup_id": rollup_id,
                    "from_address": from_address,
                    "rollup_filter": rollup_filter,
                    "window_start_utc": window_start_dt.isoformat(),
                    "window_end_exclusive_utc": window_end_exclusive_dt.isoformat(),
                    "page": page,
                    "page_size": stored_page_size,
                    "requested_page_size": page_size,
                    "relative_path": str(page_path.relative_to(repo_root())),
                    "fetched_at_utc": None,
                    "reused_existing": True,
                }
            )
            total_rows.extend(normalized_rows)
            if len(normalized_rows) < stored_page_size:
                break
            page += 1
            continue

        if cached_page_size is not None and cached_page_size != page_size:
            remaining_end_exclusive_dt = (
                total_rows[-1].timestamp_utc + timedelta(seconds=1) if total_rows else window_end_exclusive_dt
            )
            if remaining_end_exclusive_dt <= window_start_dt:
                return total_rows
            logging.info(
                "Resuming Blobscan window for %s/%s with page size %s after cached page size %s within %s..%s ending at %s",
                rollup_id,
                source_dir,
                page_size,
                cached_page_size,
                window_start_dt.isoformat(),
                window_end_exclusive_dt.isoformat(),
                remaining_end_exclusive_dt.isoformat(),
            )
            continuation_rows = fetch_blobscan_window(
                snapshot_dir=snapshot_dir,
                rollup_id=rollup_id,
                start_day=window_start_dt.date(),
                end_day_exclusive=remaining_end_exclusive_dt.date(),
                page_size=page_size,
                retries=retries,
                timeout_seconds=timeout_seconds,
                request_log=request_log,
                from_address=from_address,
                rollup_filter=rollup_filter,
                start_timestamp_utc=window_start_dt,
                end_timestamp_exclusive_utc=remaining_end_exclusive_dt,
                allow_rollup_discovery_fallback=allow_rollup_discovery_fallback,
                subtype=subtype,
            )
            merged_rows = list(total_rows)
            seen_hashes = {row.hash for row in merged_rows}
            for row in continuation_rows:
                if row.hash in seen_hashes:
                    continue
                seen_hashes.add(row.hash)
                merged_rows.append(row)
            return merged_rows

        remaining_end_exclusive_dt = total_rows[-1].timestamp_utc + timedelta(seconds=1) if total_rows else window_end_exclusive_dt
        if remaining_end_exclusive_dt <= window_start_dt:
            return total_rows
        completed_rows = maybe_reuse_completed_blobscan_scope(
            snapshot_dir=snapshot_dir,
            rollup_id=rollup_id,
            source_dir=source_dir,
            from_address=from_address,
            rollup_filter=rollup_filter,
            window_start_dt=window_start_dt,
            window_end_exclusive_dt=remaining_end_exclusive_dt,
            retries=retries,
            timeout_seconds=timeout_seconds,
        )
        if completed_rows is not None:
            merged_rows = list(total_rows)
            seen_hashes = {row.hash for row in merged_rows}
            for row in completed_rows:
                if row.hash in seen_hashes:
                    continue
                seen_hashes.add(row.hash)
                merged_rows.append(row)
            return merged_rows

        if total_rows and cached_page_size == page_size and page_size >= BLOBSCAN_TX_PAGE_SIZE:
            logging.info(
                "Skipping another live Blobscan page fetch for %s/%s within %s..%s because the exact cached scope "
                "already ends on a full final page at page size %s; rebuilding the exact scope via Blockscout "
                "txlist plus receipts instead",
                rollup_id,
                source_dir,
                window_start_dt.isoformat(),
                remaining_end_exclusive_dt.isoformat(),
                page_size,
            )
            fallback_rows = backfill_blobscan_window_from_blockscout_receipts(
                snapshot_dir=snapshot_dir,
                rollup_id=rollup_id,
                from_address=from_address,
                rollup_filter=rollup_filter,
                window_start_dt=window_start_dt,
                window_end_exclusive_dt=remaining_end_exclusive_dt,
                page_size=page_size,
                retries=retries,
                timeout_seconds=timeout_seconds,
                request_log=request_log,
            )
            if fallback_rows is not None:
                return fallback_rows

        params: dict[str, str] = {
            "startDate": iso_utc_datetime(window_start_dt),
            "endDate": iso_utc_datetime(window_end_inclusive_dt),
            "ps": str(page_size),
            "p": str(page),
        }
        if from_address:
            params["from"] = from_address
        if rollup_filter:
            params["rollups"] = blobscan_rollup_filter_value(rollup_filter)
            params["categories"] = "rollup"
        url = f"{BLOBSCAN_TX_URL}?{urlencode(params)}"
        try:
            result = fetch_json(
                url,
                headers=BLOBSCAN_HEADERS,
                retries=retries,
                timeout_seconds=timeout_seconds,
            )
        except SystemExit:
            remaining_end_exclusive_dt = (
                total_rows[-1].timestamp_utc + timedelta(seconds=1) if total_rows else window_end_exclusive_dt
            )
            remaining_span_seconds = int((remaining_end_exclusive_dt - window_start_dt).total_seconds())
            completed_rows = prove_cached_blobscan_scope_complete(
                snapshot_dir=snapshot_dir,
                rollup_id=rollup_id,
                source_dir=source_dir,
                from_address=from_address,
                rollup_filter=rollup_filter,
                window_start_dt=window_start_dt,
                window_end_exclusive_dt=remaining_end_exclusive_dt,
                retries=retries,
                timeout_seconds=timeout_seconds,
            )
            if completed_rows is not None:
                merged_rows = list(total_rows)
                seen_hashes = {row.hash for row in merged_rows}
                for row in completed_rows:
                    if row.hash in seen_hashes:
                        continue
                    seen_hashes.add(row.hash)
                    merged_rows.append(row)
                return merged_rows
            fallback_rows = backfill_blobscan_window_from_blockscout_receipts(
                snapshot_dir=snapshot_dir,
                rollup_id=rollup_id,
                from_address=from_address,
                rollup_filter=rollup_filter,
                window_start_dt=window_start_dt,
                window_end_exclusive_dt=remaining_end_exclusive_dt,
                page_size=page_size,
                retries=retries,
                timeout_seconds=timeout_seconds,
                request_log=request_log,
            )
            if fallback_rows is not None:
                merged_rows = list(total_rows)
                seen_hashes = {row.hash for row in merged_rows}
                for row in fallback_rows:
                    if row.hash in seen_hashes:
                        continue
                    seen_hashes.add(row.hash)
                    merged_rows.append(row)
                return merged_rows
            if instability_retries_remaining > 0:
                if not total_rows and page == 1:
                    logging.info(
                        "Cooling down Blobscan window for %s/%s at page size %s after repeated page 1 instability "
                        "within %s..%s; retrying the same window in %.1fs (%s retries remaining after this)",
                        rollup_id,
                        source_dir,
                        page_size,
                        window_start_dt.isoformat(),
                        remaining_end_exclusive_dt.isoformat(),
                        BLOBSCAN_INSTABILITY_RETRY_DELAY_SECONDS,
                        instability_retries_remaining - 1,
                    )
                else:
                    logging.info(
                        "Cooling down Blobscan window for %s/%s after repeated page %s instability at page size %s "
                        "within %s..%s; retrying the same partially-cached window in %.1fs (%s retries remaining after this)",
                        rollup_id,
                        source_dir,
                        page,
                        page_size,
                        window_start_dt.isoformat(),
                        remaining_end_exclusive_dt.isoformat(),
                        BLOBSCAN_INSTABILITY_RETRY_DELAY_SECONDS,
                        instability_retries_remaining - 1,
                    )
                time.sleep(BLOBSCAN_INSTABILITY_RETRY_DELAY_SECONDS)
                return fetch_blobscan_window(
                    snapshot_dir=snapshot_dir,
                    rollup_id=rollup_id,
                    start_day=window_start_dt.date(),
                    end_day_exclusive=remaining_end_exclusive_dt.date(),
                    page_size=page_size,
                    retries=retries,
                    timeout_seconds=timeout_seconds,
                    request_log=request_log,
                    from_address=from_address,
                    rollup_filter=rollup_filter,
                    start_timestamp_utc=window_start_dt,
                    end_timestamp_exclusive_utc=remaining_end_exclusive_dt,
                    instability_retries_remaining=instability_retries_remaining - 1,
                    allow_rollup_discovery_fallback=allow_rollup_discovery_fallback,
                    subtype=subtype,
                )
            if (
                not total_rows
                and page == 1
                and remaining_span_seconds <= 1
                and terminal_window_retries_remaining > 0
            ):
                logging.info(
                    "Retrying exact Blobscan window for %s/%s within %s..%s after repeated terminal instability; "
                    "retrying the same one-second scope in %.1fs (%s retries remaining after this)",
                    rollup_id,
                    source_dir,
                    window_start_dt.isoformat(),
                    remaining_end_exclusive_dt.isoformat(),
                    BLOBSCAN_TERMINAL_WINDOW_RETRY_DELAY_SECONDS,
                    terminal_window_retries_remaining - 1,
                )
                time.sleep(BLOBSCAN_TERMINAL_WINDOW_RETRY_DELAY_SECONDS)
                return fetch_blobscan_window(
                    snapshot_dir=snapshot_dir,
                    rollup_id=rollup_id,
                    start_day=window_start_dt.date(),
                    end_day_exclusive=remaining_end_exclusive_dt.date(),
                    page_size=page_size,
                    retries=retries,
                    timeout_seconds=timeout_seconds,
                    request_log=request_log,
                    from_address=from_address,
                    rollup_filter=rollup_filter,
                    start_timestamp_utc=window_start_dt,
                    end_timestamp_exclusive_utc=remaining_end_exclusive_dt,
                    instability_retries_remaining=BLOBSCAN_INSTABILITY_RETRY_ROUNDS,
                    terminal_window_retries_remaining=terminal_window_retries_remaining - 1,
                    allow_rollup_discovery_fallback=allow_rollup_discovery_fallback,
                    subtype=subtype,
                )
            fallback_page_size = max(BLOBSCAN_MIN_PAGE_SIZE, page_size // 2)
            if fallback_page_size < page_size:
                logging.info(
                    "Retrying Blobscan window for %s/%s with smaller page size %s after page %s failure at page size %s within %s..%s",
                    rollup_id,
                    source_dir,
                    fallback_page_size,
                    page,
                    page_size,
                    window_start_dt.isoformat(),
                    remaining_end_exclusive_dt.isoformat(),
                )
                fallback_rows = fetch_blobscan_window(
                    snapshot_dir=snapshot_dir,
                    rollup_id=rollup_id,
                    start_day=window_start_dt.date(),
                    end_day_exclusive=remaining_end_exclusive_dt.date(),
                    page_size=fallback_page_size,
                    retries=retries,
                    timeout_seconds=timeout_seconds,
                    request_log=request_log,
                    from_address=from_address,
                    rollup_filter=rollup_filter,
                    start_timestamp_utc=window_start_dt,
                    end_timestamp_exclusive_utc=remaining_end_exclusive_dt,
                    terminal_window_retries_remaining=terminal_window_retries_remaining,
                    allow_rollup_discovery_fallback=allow_rollup_discovery_fallback,
                    subtype=subtype,
                )
                merged_rows = list(total_rows)
                seen_hashes = {row.hash for row in merged_rows}
                for row in fallback_rows:
                    if row.hash in seen_hashes:
                        continue
                    seen_hashes.add(row.hash)
                    merged_rows.append(row)
                return merged_rows
            remaining_span_seconds = int((remaining_end_exclusive_dt - window_start_dt).total_seconds())
            if remaining_span_seconds > 1:
                split_offset_seconds = max(1, remaining_span_seconds // 2)
                split_dt = window_start_dt + timedelta(seconds=split_offset_seconds)
                logging.info(
                    "Splitting slow Blobscan window for %s/%s from %s..%s after page %s failure into %s..%s and %s..%s",
                    rollup_id,
                    source_dir,
                    window_start_dt.isoformat(),
                    remaining_end_exclusive_dt.isoformat(),
                    page,
                    window_start_dt.isoformat(),
                    split_dt.isoformat(),
                    split_dt.isoformat(),
                    remaining_end_exclusive_dt.isoformat(),
                )
                older_rows = fetch_blobscan_window(
                    snapshot_dir=snapshot_dir,
                    rollup_id=rollup_id,
                    start_day=window_start_dt.date(),
                    end_day_exclusive=split_dt.date(),
                    page_size=page_size,
                    retries=retries,
                    timeout_seconds=timeout_seconds,
                    request_log=request_log,
                    from_address=from_address,
                    rollup_filter=rollup_filter,
                    start_timestamp_utc=window_start_dt,
                    end_timestamp_exclusive_utc=split_dt,
                    terminal_window_retries_remaining=terminal_window_retries_remaining,
                    allow_rollup_discovery_fallback=allow_rollup_discovery_fallback,
                    subtype=subtype,
                )
                newer_rows = fetch_blobscan_window(
                    snapshot_dir=snapshot_dir,
                    rollup_id=rollup_id,
                    start_day=split_dt.date(),
                    end_day_exclusive=remaining_end_exclusive_dt.date(),
                    page_size=page_size,
                    retries=retries,
                    timeout_seconds=timeout_seconds,
                    request_log=request_log,
                    from_address=from_address,
                    rollup_filter=rollup_filter,
                    start_timestamp_utc=split_dt,
                    end_timestamp_exclusive_utc=remaining_end_exclusive_dt,
                    terminal_window_retries_remaining=terminal_window_retries_remaining,
                    allow_rollup_discovery_fallback=allow_rollup_discovery_fallback,
                    subtype=subtype,
                )
                merged_rows = list(total_rows)
                seen_hashes = {row.hash for row in merged_rows}
                for row in newer_rows + older_rows:
                    if row.hash in seen_hashes:
                        continue
                    seen_hashes.add(row.hash)
                    merged_rows.append(row)
                return merged_rows
            raise
        payload = result.payload
        rows = payload.get("transactions")
        if not isinstance(rows, list):
            raise SystemExit(f"Blobscan payload is malformed for {rollup_id}: {payload!r}")
        normalized_rows = [
            normalize_blobscan_tx(row, default_rollup_id=rollup_id, subtype=subtype)
            for row in rows
            if isinstance(row, dict)
        ]
        total_transactions = normalize_blobscan_total_transactions(
            rollup_id=rollup_id,
            source_dir=source_dir,
            total_transactions=total_transactions,
            row_count=len(normalized_rows),
            page_size=page_size,
        )
        if (
            page == 1
            and not total_rows
            and from_address is not None
            and rollup_filter is None
            and allow_rollup_discovery_fallback
            and total_transactions is not None
            and total_transactions > 0
            and not normalized_rows
        ):
            logging.warning(
                "Blobscan returned zero tx rows for %s/%s page 1 despite total_transactions=%s; retrying the same "
                "scope via rollups=%s instead of from=%s",
                rollup_id,
                source_dir,
                total_transactions,
                rollup_id,
                from_address,
            )
            return fetch_blobscan_window(
                snapshot_dir=snapshot_dir,
                rollup_id=rollup_id,
                start_day=window_start_dt.date(),
                end_day_exclusive=window_end_exclusive_dt.date(),
                page_size=page_size,
                retries=retries,
                timeout_seconds=timeout_seconds,
                request_log=request_log,
                rollup_filter=rollup_id,
                start_timestamp_utc=window_start_dt,
                end_timestamp_exclusive_utc=window_end_exclusive_dt,
                allow_rollup_discovery_fallback=allow_rollup_discovery_fallback,
                subtype=subtype,
            )
        for row in normalized_rows:
            if row.rollup_id != rollup_id:
                raise SystemExit(
                    f"on-chain attribution is ambiguous for {rollup_id}: "
                    f"Blobscan labeled tx {row.hash} as {row.rollup_id}"
                )
        write_json(
            page_path,
            {
                "source": "blobscan_transactions",
                "rollup_id": rollup_id,
                "window_start_utc": window_start_dt.isoformat(),
                "window_end_exclusive_utc": window_end_exclusive_dt.isoformat(),
                "page": page,
                "page_size": page_size,
                "total_transactions": total_transactions,
                "fetched_at_utc": result.fetched_at_utc,
                "url": url,
                "transactions": [blobscan_tx_record(row) for row in normalized_rows],
            },
        )
        request_log.append(
            {
                "source": "blobscan_transactions",
                "rollup_id": rollup_id,
                "from_address": from_address,
                "rollup_filter": rollup_filter,
                "window_start_utc": window_start_dt.isoformat(),
                "window_end_exclusive_utc": window_end_exclusive_dt.isoformat(),
                "page": page,
                "page_size": page_size,
                "relative_path": str(page_path.relative_to(repo_root())),
                "fetched_at_utc": result.fetched_at_utc,
            }
        )
        total_rows.extend(normalized_rows)
        if len(rows) < page_size:
            break
        page += 1

    return total_rows


def chunked(values: Iterable[int | str], size: int) -> Iterable[list[int | str]]:
    chunk: list[int | str] = []
    for value in values:
        chunk.append(value)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def chunk_cache_token(values: Iterable[int | str]) -> str:
    payload = "\n".join(str(value) for value in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def block_base_fee_batch_path(
    *,
    snapshot_dir: Path,
    batch_index: int,
    block_chunk: list[int | str],
) -> Path:
    token = chunk_cache_token(block_chunk)
    return snapshot_dir / "blockscout" / "block_base_fees" / f"batch-{batch_index:04d}__{token}.json"


def receipt_batch_path(
    *,
    snapshot_dir: Path,
    batch_index: int,
    hash_chunk: list[int | str],
) -> Path:
    token = chunk_cache_token(hash_chunk)
    return snapshot_dir / "blockscout" / "receipts" / f"batch-{batch_index:04d}__{token}.json"


def fetch_block_base_fees(
    *,
    snapshot_dir: Path,
    block_numbers: Iterable[int],
    retries: int,
    timeout_seconds: float,
    batch_size: int,
    request_log: list[dict[str, Any]],
) -> dict[int, int]:
    requested_block_numbers = sorted({int(block_number) for block_number in block_numbers})
    if not requested_block_numbers:
        return {}

    known: dict[int, int] = {}
    lookup_db = block_base_fee_lookup_db_path(snapshot_dir=snapshot_dir)
    if lookup_db.exists():
        known.update(load_block_base_fees_from_lookup_db(db_path=lookup_db, block_numbers=requested_block_numbers))
        if known:
            request_log.append(
                {
                    "source": "eth_getBlockByNumber",
                    "count": len(known),
                    "relative_path": str(lookup_db.relative_to(repo_root())),
                    "fetched_at_utc": None,
                    "reused_existing": True,
                    "reused_via": "block_base_fee_lookup_sqlite",
                }
            )
            logging.info(
                "Reused block base-fee lookup DB for %s/%s requested blocks",
                len(known),
                len(requested_block_numbers),
            )

    missing_block_numbers = [block_number for block_number in requested_block_numbers if block_number not in known]
    if missing_block_numbers:
        bigquery_block_base_fees = backfill_block_base_fees_from_bigquery(
            snapshot_dir=snapshot_dir,
            block_numbers=missing_block_numbers,
            request_log=request_log,
        )
        if bigquery_block_base_fees:
            known.update(bigquery_block_base_fees)

    batch_index = 1
    for block_chunk in chunked(requested_block_numbers, batch_size):
        path = block_base_fee_batch_path(
            snapshot_dir=snapshot_dir,
            batch_index=batch_index,
            block_chunk=block_chunk,
        )
        if all(block_number in known for block_number in block_chunk):
            batch_index += 1
            continue
        if path.exists():
            batch_known = load_existing_block_fee_batch(path)
            known.update(batch_known)
            store_block_base_fees_in_lookup_db(
                db_path=lookup_db,
                block_base_fees=batch_known.items(),
            )
            request_log.append(
                {
                    "source": "eth_getBlockByNumber",
                    "batch_index": batch_index,
                    "count": len(block_chunk),
                    "relative_path": str(path.relative_to(repo_root())),
                    "fetched_at_utc": None,
                    "reused_existing": True,
                }
            )
            if all(block_number in known for block_number in block_chunk):
                batch_index += 1
                continue
            raise SystemExit(f"stored block fee batch is incomplete in resume path: {path}")

        requests = [
            {
                "id": index,
                "jsonrpc": "2.0",
                "method": "eth_getBlockByNumber",
                "params": [hex(int(block_number)), False],
            }
            for index, block_number in enumerate(block_chunk, start=1)
        ]
        body = json.dumps(requests).encode("utf-8")
        provider_labels: list[str] = ["blockscout"]
        try:
            result = fetch_eth_rpc_json(
                body=body,
                retries=retries,
                timeout_seconds=timeout_seconds,
                provider_url=BLOCKSCOUT_RPC_URL,
            )
            payload = result.payload
        except SystemExit as exc:
            logging.warning(
                "Blockscout eth_getBlockByNumber batch %s failed (%s); retrying via %s",
                batch_index,
                exc,
                ETH_FALLBACK_RPC_URL,
            )
            result = fetch_eth_rpc_json(
                body=body,
                retries=retries,
                timeout_seconds=timeout_seconds,
                provider_url=ETH_FALLBACK_RPC_URL,
            )
            payload = result.payload
            provider_labels = ["publicnode"]

        if not isinstance(payload, list):
            if provider_labels == ["blockscout"]:
                logging.warning(
                    "Blockscout eth_getBlockByNumber batch %s returned malformed payload; retrying via %s",
                    batch_index,
                    ETH_FALLBACK_RPC_URL,
                )
                result = fetch_eth_rpc_json(
                    body=body,
                    retries=retries,
                    timeout_seconds=timeout_seconds,
                    provider_url=ETH_FALLBACK_RPC_URL,
                )
                payload = result.payload
                provider_labels = ["publicnode"]
            if not isinstance(payload, list):
                raise SystemExit(f"eth_getBlockByNumber batch response is malformed: {payload!r}")

        batch_rows: list[dict[str, Any]] = []
        fallback_requests: list[dict[str, Any]] = []
        for item in payload:
            try:
                if not isinstance(item, dict):
                    raise ValueError(f"batch item is malformed: {item!r}")
                block = item.get("result")
                if not isinstance(block, dict):
                    raise ValueError(f"returned malformed result: {item!r}")
                block_number, base_fee_wei, timestamp_utc = parse_rpc_block_record(block)
            except (KeyError, TypeError, ValueError) as exc:
                if provider_labels != ["blockscout"]:
                    raise SystemExit(f"eth_getBlockByNumber result is malformed: {item!r}") from exc
                item_id = item.get("id") if isinstance(item, dict) else None
                if not isinstance(item_id, int) or item_id < 1 or item_id > len(block_chunk):
                    raise SystemExit(f"eth_getBlockByNumber batch item is malformed: {item!r}") from exc
                fallback_requests.append(requests[item_id - 1])
                logging.warning(
                    "Blockscout returned malformed block data for %s in batch %s; retrying via %s: %s",
                    requests[item_id - 1]["params"][0],
                    batch_index,
                    ETH_FALLBACK_RPC_URL,
                    exc,
                )
                continue

            known[block_number] = base_fee_wei
            batch_rows.append(
                {
                    "block_number": block_number,
                    "base_fee_per_gas_wei": str(base_fee_wei),
                    "timestamp_utc": timestamp_utc,
                }
            )

        if fallback_requests:
            fallback_body = json.dumps(fallback_requests).encode("utf-8")
            fallback_result = fetch_eth_rpc_json(
                body=fallback_body,
                retries=retries,
                timeout_seconds=timeout_seconds,
                provider_url=ETH_FALLBACK_RPC_URL,
            )
            fallback_payload = fallback_result.payload
            if not isinstance(fallback_payload, list):
                raise SystemExit(f"eth_getBlockByNumber fallback batch response is malformed: {fallback_payload!r}")
            provider_labels.append("publicnode")
            result = fallback_result
            for item in fallback_payload:
                if not isinstance(item, dict):
                    raise SystemExit(f"eth_getBlockByNumber fallback batch item is malformed: {item!r}")
                block = item.get("result")
                if not isinstance(block, dict):
                    raise SystemExit(f"eth_getBlockByNumber fallback returned malformed result: {item!r}")
                try:
                    block_number, base_fee_wei, timestamp_utc = parse_rpc_block_record(block)
                except (KeyError, TypeError, ValueError) as exc:
                    raise SystemExit(f"eth_getBlockByNumber fallback result is malformed: {block!r}") from exc
                known[block_number] = base_fee_wei
                batch_rows.append(
                    {
                        "block_number": block_number,
                        "base_fee_per_gas_wei": str(base_fee_wei),
                        "timestamp_utc": timestamp_utc,
                    }
                )

        write_json(
            path,
            {
                "source": "eth_getBlockByNumber",
                "provider_labels": provider_labels,
                "fetched_at_utc": result.fetched_at_utc,
                "requested_block_numbers": [int(block_number) for block_number in block_chunk],
                "blocks": batch_rows,
            },
        )
        store_block_base_fees_in_lookup_db(
            db_path=lookup_db,
            block_base_fees=((int(row["block_number"]), int(str(row["base_fee_per_gas_wei"]))) for row in batch_rows),
        )
        request_log.append(
            {
                "source": "eth_getBlockByNumber",
                "batch_index": batch_index,
                "count": len(block_chunk),
                "relative_path": str(path.relative_to(repo_root())),
                "fetched_at_utc": result.fetched_at_utc,
            }
        )
        batch_index += 1

    return known


def fetch_receipts(
    *,
    snapshot_dir: Path,
    tx_hashes: Iterable[str],
    retries: int,
    timeout_seconds: float,
    batch_size: int,
    request_log: list[dict[str, Any]],
    refresh_hashes: Iterable[str] | None = None,
) -> dict[str, ReceiptFields]:
    requested_hashes = [str(tx_hash).strip().lower() for tx_hash in sorted(set(tx_hashes))]
    refresh_hash_set = {
        str(tx_hash).strip().lower()
        for tx_hash in (refresh_hashes or [])
        if str(tx_hash).strip().lower() in requested_hashes
    }
    lookup_db = receipt_lookup_db_path(snapshot_dir=snapshot_dir)
    receipts: dict[str, ReceiptFields] = {}
    if lookup_db.exists():
        lookup_receipts = load_receipts_from_lookup_db(db_path=lookup_db, tx_hashes=requested_hashes)
        receipts = {
            tx_hash: receipt
            for tx_hash, receipt in lookup_receipts.items()
            if tx_hash not in refresh_hash_set
        }
        missing_hashes = [tx_hash for tx_hash in requested_hashes if tx_hash not in receipts]
        if not missing_hashes and not refresh_hash_set:
            logging.info(
                "Reusing receipt enrichment for %s tx hashes from %s",
                len(requested_hashes),
                lookup_db.relative_to(repo_root()),
            )
            request_log.append(
                {
                    "source": "eth_getTransactionReceipt",
                    "count": len(requested_hashes),
                    "relative_path": str(lookup_db.relative_to(repo_root())),
                    "fetched_at_utc": None,
                    "reused_existing": True,
                    "reused_via": "hash_lookup_sqlite",
                }
            )
            return receipts
        if refresh_hash_set:
            logging.info(
                "Refreshing %s/%s requested receipt hashes via BigQuery or live RPC instead of trusting lookup cache %s",
                len(refresh_hash_set),
                len(requested_hashes),
                lookup_db.relative_to(repo_root()),
            )
        logging.warning(
            "Receipt lookup %s is missing %s/%s requested tx hashes; attempting BigQuery backfill before legacy batch cache",
            lookup_db,
            len(missing_hashes),
            len(requested_hashes),
        )
    else:
        missing_hashes = requested_hashes
        logging.info(
            "Receipt lookup %s does not exist yet; attempting BigQuery backfill before legacy batch cache",
            lookup_db,
        )

    if missing_hashes:
        bigquery_receipts = backfill_receipts_from_bigquery(
            snapshot_dir=snapshot_dir,
            tx_hashes=missing_hashes,
            request_log=request_log,
        )
        receipts.update(bigquery_receipts)
        missing_hashes = [tx_hash for tx_hash in requested_hashes if tx_hash not in receipts]
        if not missing_hashes:
            logging.info(
                "Reusing receipt enrichment for %s tx hashes after BigQuery backfill into %s",
                len(requested_hashes),
                lookup_db.relative_to(repo_root()),
            )
            return receipts
        logging.warning(
            "BigQuery receipt backfill still left %s/%s requested tx hashes missing; falling back to legacy batch cache",
            len(missing_hashes),
            len(requested_hashes),
        )

    batch_index = 1
    total_batches = (len(requested_hashes) + batch_size - 1) // batch_size
    for hash_chunk in chunked(requested_hashes, batch_size):
        path = receipt_batch_path(
            snapshot_dir=snapshot_dir,
            batch_index=batch_index,
            hash_chunk=hash_chunk,
        )
        if path.exists():
            for row in load_existing_receipt_batch(path):
                receipts[row.hash] = row
            request_log.append(
                {
                    "source": "eth_getTransactionReceipt",
                    "batch_index": batch_index,
                    "count": len(hash_chunk),
                    "relative_path": str(path.relative_to(repo_root())),
                    "fetched_at_utc": None,
                    "reused_existing": True,
                }
            )
            batch_index += 1
            continue

        if (
            batch_index == 1
            or batch_index == total_batches
            or batch_index % RECEIPT_FALLBACK_PROGRESS_LOG_INTERVAL == 0
        ):
            logging.info(
                "Legacy receipt batch cache progress %s/%s (%s/%s receipts cached or fetched so far)",
                batch_index,
                total_batches,
                len(receipts),
                len(requested_hashes),
            )

        chunk_missing_hashes = [tx_hash for tx_hash in hash_chunk if tx_hash not in receipts]
        if not chunk_missing_hashes:
            request_log.append(
                {
                    "source": "eth_getTransactionReceipt",
                    "batch_index": batch_index,
                    "count": len(hash_chunk),
                    "relative_path": str(lookup_db.relative_to(repo_root())),
                    "fetched_at_utc": None,
                    "reused_existing": True,
                    "reused_via": "hash_lookup_sqlite",
                }
            )
            batch_index += 1
            continue

        requests = [
            {
                "id": index,
                "jsonrpc": "2.0",
                "method": "eth_getTransactionReceipt",
                "params": [tx_hash],
            }
            for index, tx_hash in enumerate(chunk_missing_hashes, start=1)
        ]
        body = json.dumps(requests).encode("utf-8")
        provider_labels: list[str] = ["blockscout"]
        try:
            result = fetch_eth_rpc_json(
                body=body,
                retries=retries,
                timeout_seconds=timeout_seconds,
                provider_url=BLOCKSCOUT_RPC_URL,
            )
            payload = result.payload
        except SystemExit as exc:
            logging.warning(
                "Blockscout eth_getTransactionReceipt batch %s failed (%s); retrying via %s",
                batch_index,
                exc,
                ETH_FALLBACK_RPC_URL,
            )
            result = fetch_eth_rpc_json(
                body=body,
                retries=retries,
                timeout_seconds=timeout_seconds,
                provider_url=ETH_FALLBACK_RPC_URL,
            )
            payload = result.payload
            provider_labels = ["publicnode"]

        if not isinstance(payload, list):
            if provider_labels == ["blockscout"]:
                logging.warning(
                    "Blockscout eth_getTransactionReceipt batch %s returned malformed payload; retrying via %s",
                    batch_index,
                    ETH_FALLBACK_RPC_URL,
                )
                result = fetch_eth_rpc_json(
                    body=body,
                    retries=retries,
                    timeout_seconds=timeout_seconds,
                    provider_url=ETH_FALLBACK_RPC_URL,
                )
                payload = result.payload
                provider_labels = ["publicnode"]
            if not isinstance(payload, list):
                raise SystemExit(f"eth_getTransactionReceipt batch response is malformed: {payload!r}")

        compact_rows: list[dict[str, Any]] = []
        fallback_requests: list[dict[str, Any]] = []
        for item in payload:
            try:
                if not isinstance(item, dict):
                    raise ValueError(f"batch item is malformed: {item!r}")
                receipt = item.get("result")
                if not isinstance(receipt, dict):
                    raise ValueError(f"returned malformed result: {item!r}")
                row = parse_rpc_receipt(receipt)
            except (KeyError, TypeError, ValueError) as exc:
                if provider_labels != ["blockscout"]:
                    raise SystemExit(f"eth_getTransactionReceipt result is malformed: {item!r}") from exc
                item_id = item.get("id") if isinstance(item, dict) else None
                if not isinstance(item_id, int) or item_id < 1 or item_id > len(chunk_missing_hashes):
                    raise SystemExit(f"eth_getTransactionReceipt batch item is malformed: {item!r}") from exc
                fallback_requests.append(requests[item_id - 1])
                logging.warning(
                    "Blockscout returned malformed receipt for %s in batch %s; retrying via %s: %s",
                    requests[item_id - 1]["params"][0],
                    batch_index,
                    ETH_FALLBACK_RPC_URL,
                    exc,
                )
                continue

            receipts[row.hash] = row

        if fallback_requests:
            fallback_body = json.dumps(fallback_requests).encode("utf-8")
            fallback_result = fetch_eth_rpc_json(
                body=fallback_body,
                retries=retries,
                timeout_seconds=timeout_seconds,
                provider_url=ETH_FALLBACK_RPC_URL,
            )
            fallback_payload = fallback_result.payload
            if not isinstance(fallback_payload, list):
                raise SystemExit(
                    f"eth_getTransactionReceipt fallback batch response is malformed: {fallback_payload!r}"
                )
            provider_labels.append("publicnode")
            result = fallback_result
            for item in fallback_payload:
                if not isinstance(item, dict):
                    raise SystemExit(f"eth_getTransactionReceipt fallback batch item is malformed: {item!r}")
                receipt = item.get("result")
                if not isinstance(receipt, dict):
                    raise SystemExit(f"eth_getTransactionReceipt fallback returned malformed result: {item!r}")
                try:
                    row = parse_rpc_receipt(receipt)
                except (KeyError, TypeError, ValueError) as exc:
                    raise SystemExit(f"eth_getTransactionReceipt fallback result is malformed: {receipt!r}") from exc

                receipts[row.hash] = row

        compact_rows = [receipt_record_payload(receipts[tx_hash]) for tx_hash in hash_chunk if tx_hash in receipts]

        write_json(
            path,
            {
                "source": "eth_getTransactionReceipt",
                "provider_labels": provider_labels,
                "fetched_at_utc": result.fetched_at_utc,
                "requested_hashes": list(hash_chunk),
                "receipts": compact_rows,
            },
        )
        store_receipts_in_lookup_db(
            db_path=lookup_db,
            receipts=[receipts[tx_hash] for tx_hash in hash_chunk if tx_hash in receipts],
        )
        request_log.append(
            {
                "source": "eth_getTransactionReceipt",
                "batch_index": batch_index,
                "count": len(hash_chunk),
                "relative_path": str(path.relative_to(repo_root())),
                "fetched_at_utc": result.fetched_at_utc,
            }
        )
        batch_index += 1

    return receipts


def missing_receipt_hashes_from_checkpoint(
    *,
    checkpoint_path: Path,
    lookup_db: Path,
    include_calldata: bool = False,
) -> list[str]:
    connection = sqlite3.connect(f"file:{checkpoint_path}?mode=ro", uri=True)
    try:
        connection.row_factory = sqlite3.Row
        requested_hashes_cte = " UNION ".join(receipt_hash_scope_selects(include_calldata=include_calldata))
        if lookup_db.exists():
            connection.execute("ATTACH DATABASE ? AS receipts_db", (str(lookup_db),))
            query = (
                "WITH requested_hashes AS ("
                f"  {requested_hashes_cte}"
                ") "
                "SELECT h.hash "
                "FROM requested_hashes h "
                "LEFT JOIN receipts_db.receipts r ON r.hash = h.hash "
                "WHERE r.hash IS NULL "
                "ORDER BY h.hash"
            )
        else:
            query = (
                "WITH requested_hashes AS ("
                f"  {requested_hashes_cte}"
                ") "
                "SELECT hash FROM requested_hashes ORDER BY hash"
            )
        return [str(row["hash"]).strip().lower() for row in connection.execute(query)]
    finally:
        connection.close()


def count_receipt_hashes_from_checkpoint(
    *,
    checkpoint_path: Path,
    include_calldata: bool = False,
) -> int:
    connection = sqlite3.connect(f"file:{checkpoint_path}?mode=ro", uri=True)
    try:
        requested_hashes_cte = " UNION ".join(receipt_hash_scope_selects(include_calldata=include_calldata))
        query = (
            "WITH requested_hashes AS ("
            f"  {requested_hashes_cte}"
            ") "
            "SELECT COUNT(*) FROM requested_hashes"
        )
        return int(connection.execute(query).fetchone()[0])
    finally:
        connection.close()


def missing_block_numbers_from_checkpoint(
    *,
    checkpoint_path: Path,
    lookup_db: Path,
) -> list[int]:
    connection = sqlite3.connect(f"file:{checkpoint_path}?mode=ro", uri=True)
    try:
        connection.row_factory = sqlite3.Row
        if lookup_db.exists():
            connection.execute("ATTACH DATABASE ? AS fees_db", (str(lookup_db),))
            query = (
                "WITH missing_blocks AS ("
                "  SELECT DISTINCT c.block_number AS block_number "
                "  FROM calldata_txs c "
                "  LEFT JOIN fees_db.block_base_fees f ON f.block_number = c.block_number "
                "  WHERE f.block_number IS NULL "
                "  UNION "
                "  SELECT DISTINCT b.block_number AS block_number "
                "  FROM blob_txs b "
                "  LEFT JOIN fees_db.block_base_fees f ON f.block_number = b.block_number "
                "  WHERE f.block_number IS NULL"
                ") "
                "SELECT block_number FROM missing_blocks ORDER BY block_number"
            )
        else:
            query = (
                "WITH all_blocks AS ("
                "  SELECT block_number FROM calldata_txs "
                "  UNION "
                "  SELECT block_number FROM blob_txs"
                ") "
                "SELECT block_number FROM all_blocks ORDER BY block_number"
            )
        return [int(row["block_number"]) for row in connection.execute(query)]
    finally:
        connection.close()


def aggregate_daily_from_checkpoint_streaming(
    *,
    checkpoint_path: Path,
    receipt_lookup_db: Path,
    block_base_fee_lookup_db: Path,
) -> tuple[dict[tuple[str, str], dict[str, int]], dict[str, dict[str, int]], int]:
    if not receipt_lookup_db.exists():
        raise SystemExit(f"required receipt lookup DB is missing for checkpoint aggregation: {receipt_lookup_db}")
    if not block_base_fee_lookup_db.exists():
        raise SystemExit(f"required block base-fee lookup DB is missing for checkpoint aggregation: {block_base_fee_lookup_db}")

    rollup_daily: dict[tuple[str, str], dict[str, int]] = {}
    ecosystem_daily: dict[str, dict[str, int]] = {}

    def ensure_rollup_bucket(day: str, rollup_id: str) -> dict[str, int]:
        return rollup_daily.setdefault((day, rollup_id), new_rollup_component_bucket())

    def ensure_ecosystem_bucket(day: str) -> dict[str, int]:
        return ecosystem_daily.setdefault(
            day,
            {
                "base_fee_burn_wei": 0,
                "blob_fee_burn_wei": 0,
                "priority_fee_wei": 0,
                "blob_gas_used": 0,
                "calldata_gas_proxy": 0,
            },
        )

    connection = sqlite3.connect(f"file:{checkpoint_path}?mode=ro", uri=True)
    try:
        connection.row_factory = sqlite3.Row
        connection.execute("ATTACH DATABASE ? AS receipts_db", (str(receipt_lookup_db),))
        connection.execute("ATTACH DATABASE ? AS fees_db", (str(block_base_fee_lookup_db),))

        dropped_funding_like_txs = int(
            connection.execute(
                "SELECT COUNT(*) "
                "FROM calldata_txs c "
                "LEFT JOIN receipts_db.receipts r ON r.hash = c.hash "
                "WHERE c.value_wei != '0' AND COALESCE(r.gas_used, c.gas_used) = 21000"
            ).fetchone()[0]
        )

        calldata_query = (
            "SELECT substr(c.timestamp_utc, 1, 10) AS day, "
            "c.rollup_id AS rollup_id, "
            "c.subtype AS subtype, "
            "COALESCE(r.gas_used, c.gas_used) AS gas_used, "
            "COALESCE(r.effective_gas_price_wei, c.gas_price_wei) AS effective_gas_price_wei, "
            "COALESCE(r.blob_gas_used, 0) AS blob_gas_used, "
            "COALESCE(r.blob_gas_price_wei, 0) AS blob_gas_price_wei, "
            "f.base_fee_per_gas_wei AS base_fee_per_gas_wei "
            "FROM calldata_txs c "
            "LEFT JOIN receipts_db.receipts r ON r.hash = c.hash "
            "JOIN fees_db.block_base_fees f ON f.block_number = c.block_number "
            "WHERE NOT (c.value_wei != '0' AND COALESCE(r.gas_used, c.gas_used) = 21000)"
        )
        for row in connection.execute(calldata_query):
            gas_used = int(row["gas_used"])
            base_fee_wei = int(str(row["base_fee_per_gas_wei"]))
            effective_gas_price_wei = int(str(row["effective_gas_price_wei"]))
            blob_gas_used = int(row["blob_gas_used"])
            blob_gas_price_wei = int(str(row["blob_gas_price_wei"]))
            priority_fee_wei = gas_used * max(effective_gas_price_wei - base_fee_wei, 0)
            base_burn_wei = gas_used * base_fee_wei
            blob_fee_wei = blob_gas_used * blob_gas_price_wei if blob_gas_used > 0 else 0
            day = str(row["day"])
            rollup_id = str(row["rollup_id"])
            subtype = str(row["subtype"])

            if not is_canonical_rollup_subtype_in_scope(rollup_id=rollup_id, subtype=subtype):
                continue

            rollup_bucket = ensure_rollup_bucket(day, rollup_id)
            add_rollup_fee_components(
                rollup_bucket,
                subtype=subtype,
                base_burn_wei=base_burn_wei,
                priority_fee_wei=priority_fee_wei,
                blob_fee_wei=blob_fee_wei,
            )

            ecosystem_bucket = ensure_ecosystem_bucket(day)
            ecosystem_bucket["base_fee_burn_wei"] += base_burn_wei
            ecosystem_bucket["priority_fee_wei"] += priority_fee_wei
            ecosystem_bucket["calldata_gas_proxy"] += gas_used
            if blob_gas_used > 0:
                ecosystem_bucket["blob_fee_burn_wei"] += blob_fee_wei
                ecosystem_bucket["blob_gas_used"] += blob_gas_used

        blob_query = (
            "SELECT substr(b.timestamp_utc, 1, 10) AS day, "
            "b.rollup_id AS rollup_id, "
            "b.subtype AS subtype, "
            "b.blob_gas_used AS blob_gas_used, "
            "b.blob_gas_price_wei AS blob_gas_price_wei, "
            "b.blob_as_calldata_gas_used AS blob_as_calldata_gas_used, "
            "r.gas_used AS gas_used, "
            "r.effective_gas_price_wei AS effective_gas_price_wei, "
            "f.base_fee_per_gas_wei AS base_fee_per_gas_wei "
            "FROM blob_txs b "
            "JOIN receipts_db.receipts r ON r.hash = b.hash "
            "JOIN fees_db.block_base_fees f ON f.block_number = b.block_number"
        )
        for row in connection.execute(blob_query):
            gas_used = int(row["gas_used"])
            base_fee_wei = int(str(row["base_fee_per_gas_wei"]))
            effective_gas_price_wei = int(str(row["effective_gas_price_wei"]))
            blob_gas_used = int(row["blob_gas_used"])
            blob_gas_price_wei = int(str(row["blob_gas_price_wei"]))
            blob_as_calldata_gas_used = int(row["blob_as_calldata_gas_used"])
            priority_fee_wei = gas_used * max(effective_gas_price_wei - base_fee_wei, 0)
            base_burn_wei = gas_used * base_fee_wei
            blob_fee_wei = blob_gas_used * blob_gas_price_wei
            day = str(row["day"])
            rollup_id = str(row["rollup_id"])
            subtype = str(row["subtype"])

            if not is_canonical_rollup_subtype_in_scope(rollup_id=rollup_id, subtype=subtype):
                continue

            rollup_bucket = ensure_rollup_bucket(day, rollup_id)
            add_rollup_fee_components(
                rollup_bucket,
                subtype=subtype,
                base_burn_wei=base_burn_wei,
                priority_fee_wei=priority_fee_wei,
                blob_fee_wei=blob_fee_wei,
            )

            ecosystem_bucket = ensure_ecosystem_bucket(day)
            ecosystem_bucket["base_fee_burn_wei"] += base_burn_wei
            ecosystem_bucket["blob_fee_burn_wei"] += blob_fee_wei
            ecosystem_bucket["priority_fee_wei"] += priority_fee_wei
            ecosystem_bucket["blob_gas_used"] += blob_gas_used
            ecosystem_bucket["calldata_gas_proxy"] += blob_as_calldata_gas_used
    finally:
        connection.close()

    return rollup_daily, ecosystem_daily, dropped_funding_like_txs


def to_decimal_eth(value_wei: int) -> Decimal:
    return Decimal(value_wei) / WEI_PER_ETH


def format_decimal(value: Decimal) -> str:
    text = format(value.normalize(), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text if text not in {"", "-0"} else "0"


def build_raw_manifest(
    *,
    source: str,
    snapshot_dir: Path,
    command: str,
    as_of: date,
) -> dict[str, Any]:
    receipts_lookup_path = snapshot_dir / "blockscout" / "receipts_hash_lookup.sqlite3"
    block_base_fees_lookup_path = snapshot_dir / "blockscout" / "block_base_fees_lookup.sqlite3"

    def include_in_raw_manifest(path: Path) -> bool:
        rel = path.relative_to(snapshot_dir)
        parts = rel.parts
        name = rel.name

        # Runtime checkpoints and resume logs are operational artifacts, not canonical raw inputs.
        if parts and parts[0] == "_runtime":
            return False
        if name.startswith("resume_") and name.endswith(".log"):
            return False
        if name.endswith((".sqlite3-journal", ".sqlite3-shm", ".sqlite3-wal")):
            return False

        # Once the lookup DB exists, the legacy batch shards are redundant provenance noise.
        if (
            len(parts) >= 2
            and parts[0] == "blockscout"
            and parts[1] == "receipts"
            and name.startswith("batch-")
            and name.endswith(".json")
            and receipts_lookup_path.exists()
        ):
            return False
        if (
            len(parts) >= 2
            and parts[0] == "blockscout"
            and parts[1] == "block_base_fees"
            and name.startswith("batch-")
            and name.endswith(".json")
            and block_base_fees_lookup_path.exists()
        ):
            return False

        return True

    files: list[dict[str, Any]] = []
    for path in sorted(snapshot_dir.rglob("*")):
        if not path.is_file():
            continue
        if not include_in_raw_manifest(path):
            continue
        files.append(
            {
                "path": str(path.relative_to(repo_root())),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )

    return {
        "source": source,
        "as_of_utc_date": as_of.isoformat(),
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "files": files,
        "environment": {
            "python_version": sys.version.split()[0],
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
    }


def load_vendor_panel(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"required growthepie vendor panel is missing: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_registry_attribution_inputs(
    *,
    rollups: list[RegistryRollup],
    vendor_rows: list[dict[str, str]],
    observed_end: date,
    tracked_transactions_by_rollup: dict[str, dict[str, list[TrackedFunctionCall]]],
) -> None:
    vendor_pre_dencun_counts: dict[str, int] = {}
    for row in vendor_rows:
        rollup_id = normalize_slug(row.get("rollup_id", ""))
        if not rollup_id:
            continue
        if row.get("date_utc", "") >= DENCUN_DATE.isoformat():
            continue
        vendor_pre_dencun_counts[rollup_id] = vendor_pre_dencun_counts.get(rollup_id, 0) + 1

    missing_pre_dencun: list[str] = []
    for rollup in rollups:
        active_start = max(PROTOCOL_START, rollup.start_date_utc)
        active_end = observed_end if rollup.end_date_utc is None else min(observed_end, rollup.end_date_utc)
        pre_dencun_end = min(active_end, DENCUN_DATE - timedelta(days=1))
        if pre_dencun_end < active_start:
            continue
        if rollup.batcher_addresses:
            continue
        tracked_transactions = tracked_transactions_by_rollup.get(rollup.rollup_id, {})
        if relevant_pre_dencun_tracked_calls(
            rollup=rollup,
            tracked_transactions=tracked_transactions,
            active_start=active_start,
            pre_dencun_end=pre_dencun_end,
        ):
            continue

        coverage_days = vendor_pre_dencun_counts.get(rollup.rollup_id, 0)
        coverage_suffix = f", growthepie_pre_dencun_rows={coverage_days}" if coverage_days else ""
        missing_pre_dencun.append(
            f"{rollup.rollup_id}[active_pre_dencun={active_start.isoformat()}..{pre_dencun_end.isoformat()}{coverage_suffix}]"
        )

    if missing_pre_dencun:
        raise SystemExit(
            "required registry attribution inputs are missing for pre-Dencun rollups: "
            + "; ".join(missing_pre_dencun)
        )


def write_csv(path: Path, rows: list[dict[str, str]], *, headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def sample_rows_or_die(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_key = {(row["date_utc"], row["rollup_id"]): row for row in rows}
    sample: list[dict[str, str]] = []
    for day in SAMPLE_DATES:
        for rollup_id in SAMPLE_ROLLUPS:
            key = (day, rollup_id)
            if key not in by_key:
                raise SystemExit(f"sample selection missing required panel row: {key}")
            sample.append(by_key[key])
    return sample


def sample_decomp_rows_or_die(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_day = {row["date_utc"]: row for row in rows}
    sample: list[dict[str, str]] = []
    for day in SAMPLE_DATES:
        if day not in by_day:
            raise SystemExit(f"sample selection missing required decomposition row: {day}")
        sample.append(by_day[day])
    return sample


def build_processed_manifest(
    *,
    root: Path,
    run_date: date,
    inputs: list[str],
    script_path: str,
    command: str,
    output_paths: list[Path],
) -> dict[str, Any]:
    return {
        "as_of_utc_date": run_date.isoformat(),
        "inputs": inputs,
        "transform": {
            "script_path": script_path,
            "git_sha": git_sha(root),
            "command": command,
        },
        "outputs": [
            {
                "path": str(path.relative_to(root)),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(output_paths)
        ],
    }


def coerce_int(value: str | int | None) -> int:
    if value in (None, ""):
        return 0
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except ValueError as exc:
        raise SystemExit(f"expected integer-like value, got {value!r}") from exc


def normalize_slug(value: str) -> str:
    return value.strip().lower().replace("-", "_")


def new_rollup_component_bucket() -> dict[str, int]:
    return {
        "rent_paid_wei": 0,
        "batch_submissions_wei": 0,
        "proof_submissions_wei": 0,
        "state_updates_wei": 0,
        "execution_base_fee_burn_wei": 0,
        "execution_priority_fee_wei": 0,
        "blob_fee_burn_wei": 0,
    }


def add_rollup_fee_components(
    bucket: dict[str, int],
    *,
    subtype: str,
    base_burn_wei: int,
    priority_fee_wei: int,
    blob_fee_wei: int = 0,
) -> None:
    subtype_field = ROLLUP_SUBTYPE_TO_COMPONENT_FIELD.get(subtype)
    if subtype_field is None:
        raise SystemExit(f"unknown tracked transaction subtype for rollup rent decomposition: {subtype}")

    total_rent_wei = base_burn_wei + priority_fee_wei + blob_fee_wei
    bucket["rent_paid_wei"] += total_rent_wei
    bucket[subtype_field] += total_rent_wei
    bucket["execution_base_fee_burn_wei"] += base_burn_wei
    bucket["execution_priority_fee_wei"] += priority_fee_wei
    bucket["blob_fee_burn_wei"] += blob_fee_wei


def is_canonical_rollup_subtype_in_scope(*, rollup_id: str, subtype: str) -> bool:
    excluded_subtypes = CANONICAL_EXCLUDED_SUBTYPES_BY_ROLLUP.get(rollup_id)
    return excluded_subtypes is None or subtype not in excluded_subtypes


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.rpc_batch_size < 1:
        raise SystemExit(f"--rpc-batch-size must be positive, got {args.rpc_batch_size}")
    if args.rpc_batch_size > RPC_BATCH_SIZE:
        logging.info(
            "Clamping requested Blockscout RPC batch size from %s to %s to match the public endpoint limit",
            args.rpc_batch_size,
            RPC_BATCH_SIZE,
        )
        args.rpc_batch_size = RPC_BATCH_SIZE

    root = repo_root()
    run_date = args.run_date
    observed_end = observed_end_date(run_date)
    if observed_end < PROTOCOL_START:
        raise SystemExit(f"run-date {run_date.isoformat()} is before the protocol start window")

    snapshot_dir = root / "data" / "raw" / "l1_rent" / run_date.isoformat()
    raw_manifest_path = root / "data" / "raw_manifest" / f"l1_rent_{run_date.isoformat()}.json"
    decomp_path = root / "data" / "processed" / "l1_rent" / "daily_l1_rent_decomposition.csv"
    component_path = root / "data" / "processed" / "l1_rent" / "daily_rollup_rent_components.csv"
    panel_path = root / "data" / "processed" / "panels" / "daily_rollup_panel.csv"
    decomp_sample_path = root / "data" / "samples" / "l1_rent" / "daily_l1_rent_decomposition_sample.csv"
    component_sample_path = root / "data" / "samples" / "l1_rent" / "daily_rollup_rent_components_sample.csv"
    panel_sample_path = root / "data" / "samples" / "panels" / "daily_rollup_panel_sample.csv"
    decomp_manifest_path = (
        root / "data" / "processed_manifest" / f"daily_l1_rent_decomposition_{run_date.isoformat()}.json"
    )
    component_manifest_path = (
        root / "data" / "processed_manifest" / f"daily_rollup_rent_components_{run_date.isoformat()}.json"
    )
    panel_manifest_path = root / "data" / "processed_manifest" / f"daily_rollup_panel_{run_date.isoformat()}.json"

    registry_path = root / "registry" / "rollup_registry_v1.csv"
    growthepie_raw_manifest_path = root / "data" / "raw_manifest" / f"growthepie_{run_date.isoformat()}.json"
    if not growthepie_raw_manifest_path.exists():
        raise SystemExit(f"required growthepie raw manifest is missing: {growthepie_raw_manifest_path}")
    vendor_panel_path = root / "data" / "processed" / "growthepie" / "vendor_daily_rollup_panel.csv"

    if args.resume_manifested_run:
        if not snapshot_dir.exists():
            raise SystemExit(
                f"--resume-manifested-run requires an existing raw snapshot for this run date: {snapshot_dir}"
            )
        if not raw_manifest_path.exists():
            raise SystemExit(
                f"--resume-manifested-run requires an existing raw manifest for this run date: {raw_manifest_path}"
            )
        if not decomp_manifest_path.exists() or not panel_manifest_path.exists():
            raise SystemExit(
                "--resume-manifested-run requires existing processed manifests so the replay is explicitly "
                "repairing a prior manifested run"
            )
        logging.warning(
            "Reusing manifested run-date %s for a deterministic ETL repair; raw/processed manifests will be refreshed "
            "in place after reusing the existing raw snapshot",
            run_date.isoformat(),
        )
    else:
        prepare_snapshot_dir(snapshot_dir, raw_manifest_path=raw_manifest_path)
        ensure_new_manifest(raw_manifest_path, label="raw manifest")
        ensure_new_manifest(decomp_manifest_path, label="processed decomposition manifest")
        ensure_new_manifest(component_manifest_path, label="processed rollup rent component manifest")
        ensure_new_manifest(panel_manifest_path, label="processed panel manifest")

    vendor_rows = load_vendor_panel(vendor_panel_path)
    request_log: list[dict[str, Any]] = []
    checkpoint_path: Path | None = None
    partition_checkpoint = load_partition_checkpoint_if_valid(
        snapshot_dir=snapshot_dir,
        registry_path=registry_path,
    )
    if partition_checkpoint is not None:
        checkpoint_path = partition_checkpoint.checkpoint_path
        request_log = list(partition_checkpoint.request_log)
        calldata_candidate_txs_count = partition_checkpoint.calldata_candidate_txs_count
        excluded_blob_overlap_txs = partition_checkpoint.excluded_blob_overlap_txs
        calldata_txs_count = partition_checkpoint.calldata_txs_count
        blob_txs_count = partition_checkpoint.blob_txs_count
    else:
        rollups = load_registry(registry_path)
        tracked_transactions_by_rollup: dict[str, dict[str, list[TrackedFunctionCall]]] = {}
        for rollup in rollups:
            active_start = max(PROTOCOL_START, rollup.start_date_utc)
            active_end = observed_end if rollup.end_date_utc is None else min(observed_end, rollup.end_date_utc)
            if active_end < active_start:
                continue
            try:
                tracked_transactions_by_rollup[rollup.rollup_id] = fetch_l2beat_tracked_transactions(
                    snapshot_dir=snapshot_dir,
                    rollup=rollup,
                    observed_end=active_end,
                    retries=args.retries,
                    timeout_seconds=args.timeout_seconds,
                    request_log=request_log,
                )
            except SystemExit as exc:
                if not rollup.batcher_addresses:
                    raise
                logging.warning(
                    "Supplemental L2BEAT tracked transactions are unavailable for %s; continuing with batcher-address attribution plus legacy supplements only: %s",
                    rollup.rollup_id,
                    exc,
                )
                tracked_transactions_by_rollup[rollup.rollup_id] = bound_tracked_transactions_to_observed_history(
                    snapshot_dir=snapshot_dir,
                    rollup_id=rollup.rollup_id,
                    tracked_transactions=supplement_tracked_transactions(
                        rollup_id=rollup.rollup_id,
                        tracked_transactions={},
                    ),
                    observed_end=active_end,
                    request_log=request_log,
                )

        validate_registry_attribution_inputs(
            rollups=rollups,
            vendor_rows=vendor_rows,
            observed_end=observed_end,
            tracked_transactions_by_rollup=tracked_transactions_by_rollup,
        )

        calldata_candidate_txs: dict[str, BlockscoutTx] = {}
        post_dencun_blob_txs: dict[str, BlobscanTx] = {}

        for rollup in rollups:
            active_start = max(PROTOCOL_START, rollup.start_date_utc)
            active_end = observed_end if rollup.end_date_utc is None else min(observed_end, rollup.end_date_utc)
            if active_end < active_start:
                continue
            tracked_calls = relevant_calldata_tracked_calls(
                rollup=rollup,
                tracked_transactions=tracked_transactions_by_rollup.get(rollup.rollup_id, {}),
                active_start=active_start,
                active_end=active_end,
            )

            if rollup.batcher_addresses:
                for address in rollup.batcher_addresses:
                    for window_start, window_end_exclusive in month_windows(active_start, active_end):
                        rows = fetch_blockscout_tx_window(
                            snapshot_dir=snapshot_dir,
                            rollup_id=rollup.rollup_id,
                            address=address,
                            start_day=window_start,
                            end_day_exclusive=window_end_exclusive,
                            page_size=args.blockscout_page_size,
                            retries=args.retries,
                            timeout_seconds=args.timeout_seconds,
                            request_log=request_log,
                            subtype="batchSubmissions",
                        )
                        for row in rows:
                            existing = calldata_candidate_txs.get(row.hash)
                            if existing is not None and existing.rollup_id != row.rollup_id:
                                raise SystemExit(
                                    f"on-chain attribution is ambiguous: duplicate calldata-candidate tx hash {row.hash} "
                                    f"for {existing.rollup_id} and {rollup.rollup_id}"
                                )
                            calldata_candidate_txs[row.hash] = row
            if not rollup.batcher_addresses and not tracked_calls:
                raise SystemExit(
                    "required registry attribution inputs are missing for calldata-attributed rollups: "
                    f"{rollup.rollup_id}[active={active_start.isoformat()}..{active_end.isoformat()}]"
                )
            if tracked_calls:
                for tracked_call in tracked_calls:
                    tracked_start = max(active_start, timestamp_to_utc_date(tracked_call.since_timestamp))
                    tracked_end = active_end
                    if tracked_call.until_timestamp is not None:
                        tracked_end = min(tracked_end, timestamp_to_utc_date(tracked_call.until_timestamp))
                    if tracked_end < tracked_start:
                        continue
                    selector_scope = tracked_call.selector
                    for window_start, window_end_exclusive in month_windows(tracked_start, tracked_end):
                        rows = fetch_blockscout_tx_window(
                            snapshot_dir=snapshot_dir,
                            rollup_id=rollup.rollup_id,
                            address=tracked_call.address,
                            start_day=window_start,
                            end_day_exclusive=window_end_exclusive,
                            page_size=args.blockscout_page_size,
                            retries=args.retries,
                            timeout_seconds=args.timeout_seconds,
                            request_log=request_log,
                            address_role="to",
                            method_selectors=(tracked_call.selector,),
                            path_prefix="txlist_to",
                            scope_id=selector_scope,
                            subtype=tracked_call.subtype,
                        )
                        for row in rows:
                            existing = calldata_candidate_txs.get(row.hash)
                            if existing is not None and existing.rollup_id != row.rollup_id:
                                raise SystemExit(
                                    f"on-chain attribution is ambiguous: duplicate calldata-candidate tx hash {row.hash} "
                                    f"for {existing.rollup_id} and {rollup.rollup_id}"
                                )
                            calldata_candidate_txs[row.hash] = row

            post_start = max(active_start, DENCUN_DATE)
            if active_end < post_start:
                continue
            for window_start, window_end_exclusive in month_windows(post_start, active_end):
                if rollup.batcher_addresses:
                    rollup_window_start_dt = datetime_utc_start(window_start)
                    rollup_window_end_exclusive_dt = datetime_utc_start(window_end_exclusive)
                    rollup_source_dir = f"rollup_{rollup.rollup_id}"
                    if blobscan_scope_has_cached_state(
                        snapshot_dir=snapshot_dir,
                        rollup_id=rollup.rollup_id,
                        source_dir=rollup_source_dir,
                        window_start_dt=rollup_window_start_dt,
                        window_end_exclusive_dt=rollup_window_end_exclusive_dt,
                    ):
                        rows = fetch_blobscan_window(
                            snapshot_dir=snapshot_dir,
                            rollup_id=rollup.rollup_id,
                            start_day=window_start,
                            end_day_exclusive=window_end_exclusive,
                            page_size=args.blobscan_page_size,
                            retries=args.retries,
                            timeout_seconds=args.timeout_seconds,
                            request_log=request_log,
                            rollup_filter=rollup.rollup_id,
                            subtype="batchSubmissions",
                        )
                        for row in rows:
                            existing = post_dencun_blob_txs.get(row.hash)
                            if existing is not None and existing.rollup_id != row.rollup_id:
                                raise SystemExit(
                                    f"on-chain attribution is ambiguous: duplicate Blobscan tx hash {row.hash} "
                                    f"for {existing.rollup_id} and {row.rollup_id}"
                                )
                            post_dencun_blob_txs[row.hash] = row
                        continue
                    window_blob_rows: list[BlobscanTx] = []
                    for address in rollup.batcher_addresses:
                        rows = fetch_blobscan_window(
                            snapshot_dir=snapshot_dir,
                            rollup_id=rollup.rollup_id,
                            start_day=window_start,
                            end_day_exclusive=window_end_exclusive,
                            page_size=args.blobscan_page_size,
                            retries=args.retries,
                            timeout_seconds=args.timeout_seconds,
                            request_log=request_log,
                            from_address=address,
                            # When every sender-scoped query is empty, the caller retries the
                            # whole window once via rollup filter after the loop.
                            allow_rollup_discovery_fallback=False,
                            subtype="batchSubmissions",
                        )
                        window_blob_rows.extend(rows)
                        for row in rows:
                            existing = post_dencun_blob_txs.get(row.hash)
                            if existing is not None and existing.rollup_id != row.rollup_id:
                                raise SystemExit(
                                    f"on-chain attribution is ambiguous: duplicate Blobscan tx hash {row.hash} "
                                    f"for {existing.rollup_id} and {row.rollup_id}"
                                )
                            post_dencun_blob_txs[row.hash] = row
                    if not window_blob_rows:
                        logging.warning(
                            "Blobscan address-scoped coverage is empty for %s within %s..%s; retrying the same "
                            "window via rollup filter instead of gating on unstable count-derived totals",
                            rollup.rollup_id,
                            window_start.isoformat(),
                            window_end_exclusive.isoformat(),
                        )
                        rows = fetch_blobscan_window(
                            snapshot_dir=snapshot_dir,
                            rollup_id=rollup.rollup_id,
                            start_day=window_start,
                            end_day_exclusive=window_end_exclusive,
                            page_size=args.blobscan_page_size,
                            retries=args.retries,
                            timeout_seconds=args.timeout_seconds,
                            request_log=request_log,
                            rollup_filter=rollup.rollup_id,
                            subtype="batchSubmissions",
                        )
                        for row in rows:
                            existing = post_dencun_blob_txs.get(row.hash)
                            if existing is not None and existing.rollup_id != row.rollup_id:
                                raise SystemExit(
                                    f"on-chain attribution is ambiguous: duplicate Blobscan tx hash {row.hash} "
                                    f"for {existing.rollup_id} and {row.rollup_id}"
                                )
                            post_dencun_blob_txs[row.hash] = row
                elif rollup.rollup_id in ROLLUPS_WITHOUT_BATCHER_ADDRESSES:
                    rows = fetch_blobscan_window(
                        snapshot_dir=snapshot_dir,
                        rollup_id=rollup.rollup_id,
                        start_day=window_start,
                        end_day_exclusive=window_end_exclusive,
                        page_size=args.blobscan_page_size,
                        retries=args.retries,
                        timeout_seconds=args.timeout_seconds,
                        request_log=request_log,
                        rollup_filter=rollup.rollup_id,
                        subtype="batchSubmissions",
                    )
                    for row in rows:
                        existing = post_dencun_blob_txs.get(row.hash)
                        if existing is not None and existing.rollup_id != row.rollup_id:
                            raise SystemExit(
                                f"on-chain attribution is ambiguous: duplicate Blobscan tx hash {row.hash} "
                                f"for {existing.rollup_id} and {row.rollup_id}"
                            )
                        post_dencun_blob_txs[row.hash] = row
                else:
                    raise SystemExit(f"required registry attribution inputs are missing for {rollup.rollup_id}")

        supplemental_calldata_addresses_by_rollup: dict[str, dict[str, date]] = {}
        for rollup in rollups:
            if not rollup.batcher_addresses:
                continue
            registered_addresses = {address.lower() for address in rollup.batcher_addresses}
            observed_blob_senders: dict[str, date] = {}
            for blob_tx in post_dencun_blob_txs.values():
                if blob_tx.rollup_id != rollup.rollup_id or blob_tx.from_address in registered_addresses:
                    continue
                sender_start = blob_tx.timestamp_utc.date()
                existing_start = observed_blob_senders.get(blob_tx.from_address)
                if existing_start is None or sender_start < existing_start:
                    observed_blob_senders[blob_tx.from_address] = sender_start
            if observed_blob_senders:
                supplemental_calldata_addresses_by_rollup[rollup.rollup_id] = observed_blob_senders

        for rollup in rollups:
            supplemental_addresses = supplemental_calldata_addresses_by_rollup.get(rollup.rollup_id, {})
            if not supplemental_addresses:
                continue
            active_start = max(PROTOCOL_START, rollup.start_date_utc)
            post_start = max(active_start, DENCUN_DATE)
            active_end = observed_end if rollup.end_date_utc is None else min(observed_end, rollup.end_date_utc)
            if active_end < active_start:
                continue
            logging.info(
                "Discovered %s supplemental calldata sender addresses for %s from Blobscan rollup rows: %s",
                len(supplemental_addresses),
                rollup.rollup_id,
                ", ".join(
                    f"{address} (observed from {sender_start.isoformat()})"
                    for address, sender_start in sorted(supplemental_addresses.items())
                ),
            )
            for address, sender_start in sorted(supplemental_addresses.items()):
                supplemental_start = post_start
                if active_end < supplemental_start:
                    continue
                for window_start, window_end_exclusive in month_windows(supplemental_start, active_end):
                    rows = fetch_blockscout_tx_window(
                        snapshot_dir=snapshot_dir,
                        rollup_id=rollup.rollup_id,
                        address=address,
                        start_day=window_start,
                        end_day_exclusive=window_end_exclusive,
                        page_size=args.blockscout_page_size,
                        retries=args.retries,
                        timeout_seconds=args.timeout_seconds,
                        request_log=request_log,
                        subtype="batchSubmissions",
                    )
                    for row in rows:
                        existing = calldata_candidate_txs.get(row.hash)
                        if existing is not None and existing.rollup_id != row.rollup_id:
                            raise SystemExit(
                                f"on-chain attribution is ambiguous: duplicate calldata-candidate tx hash {row.hash} "
                                f"for {existing.rollup_id} and {rollup.rollup_id}"
                            )
                        calldata_candidate_txs[row.hash] = row

        calldata_txs = {}
        excluded_blob_overlap_txs = 0
        for tx_hash, tx in calldata_candidate_txs.items():
            blob_tx = post_dencun_blob_txs.get(tx_hash)
            if blob_tx is None:
                calldata_txs[tx_hash] = tx
                continue
            if blob_tx.rollup_id != tx.rollup_id:
                raise SystemExit(
                    f"on-chain attribution is ambiguous: tx hash {tx_hash} maps to calldata candidate "
                    f"{tx.rollup_id} and Blobscan rollup {blob_tx.rollup_id}"
                )
            post_dencun_blob_txs[tx_hash] = replace(blob_tx, subtype=tx.subtype)
            excluded_blob_overlap_txs += 1

        calldata_candidate_txs_count = len(calldata_candidate_txs)
        logging.info(
            "Partitioned on-chain txs into %s calldata txs and %s blob txs after excluding %s Blobscan-overlapping "
            "hashes from %s calldata candidates",
            len(calldata_txs),
            len(post_dencun_blob_txs),
            excluded_blob_overlap_txs,
            calldata_candidate_txs_count,
        )
        write_partition_checkpoint(
            snapshot_dir=snapshot_dir,
            registry_path=registry_path,
            calldata_candidate_txs_count=calldata_candidate_txs_count,
            excluded_blob_overlap_txs=excluded_blob_overlap_txs,
            calldata_txs=calldata_txs,
            blob_txs=post_dencun_blob_txs,
            request_log=request_log,
        )
        calldata_txs_count = len(calldata_txs)
        blob_txs_count = len(post_dencun_blob_txs)

    # The authoritative rent partition is transaction-level, not day-level. Blockscout
    # supplies the calldata-candidate stream across the full active window, while
    # Blobscan identifies post-Dencun blob txs. Removing any Blobscan-overlapping hashes
    # from the Blockscout side keeps mixed-regime days correct without chain-specific
    # cutover guesses.
    if checkpoint_path is not None:
        receipt_lookup_db = receipt_lookup_db_path(snapshot_dir=snapshot_dir)
        total_receipt_hashes = count_receipt_hashes_from_checkpoint(
            checkpoint_path=checkpoint_path,
            include_calldata=True,
        )
        missing_receipt_hashes = missing_receipt_hashes_from_checkpoint(
            checkpoint_path=checkpoint_path,
            lookup_db=receipt_lookup_db,
            include_calldata=True,
        )
        if missing_receipt_hashes:
            logging.warning(
                "Checkpoint resume is missing %s/%s tx receipts; backfilling only the missing hashes",
                len(missing_receipt_hashes),
                total_receipt_hashes,
            )
            fetch_receipts(
                snapshot_dir=snapshot_dir,
                tx_hashes=missing_receipt_hashes,
                retries=args.retries,
                timeout_seconds=args.timeout_seconds,
                batch_size=args.rpc_batch_size,
                request_log=request_log,
            )
            remaining_missing_receipt_hashes = missing_receipt_hashes_from_checkpoint(
                checkpoint_path=checkpoint_path,
                lookup_db=receipt_lookup_db,
                include_calldata=True,
            )
            if remaining_missing_receipt_hashes:
                raise SystemExit(
                    f"missing receipt enrichment for {len(remaining_missing_receipt_hashes)} checkpoint tx hashes "
                    "after checkpoint resume backfill"
                )
        else:
            logging.info(
                "Reusing receipt enrichment for %s checkpoint tx hashes from %s",
                total_receipt_hashes,
                receipt_lookup_db.relative_to(repo_root()),
            )
            request_log.append(
                {
                    "source": "eth_getTransactionReceipt",
                    "count": total_receipt_hashes,
                    "relative_path": str(receipt_lookup_db.relative_to(repo_root())),
                    "fetched_at_utc": None,
                    "reused_existing": True,
                    "reused_via": "hash_lookup_sqlite_checkpoint",
                }
            )

        block_base_fee_lookup_db = block_base_fee_lookup_db_path(snapshot_dir=snapshot_dir)
        missing_block_numbers = missing_block_numbers_from_checkpoint(
            checkpoint_path=checkpoint_path,
            lookup_db=block_base_fee_lookup_db,
        )
        if missing_block_numbers:
            logging.warning(
                "Checkpoint resume is missing %s block base fees; backfilling only the missing blocks",
                len(missing_block_numbers),
            )
            fetch_block_base_fees(
                snapshot_dir=snapshot_dir,
                block_numbers=missing_block_numbers,
                retries=args.retries,
                timeout_seconds=args.timeout_seconds,
                batch_size=args.rpc_batch_size,
                request_log=request_log,
            )
            remaining_missing_block_numbers = missing_block_numbers_from_checkpoint(
                checkpoint_path=checkpoint_path,
                lookup_db=block_base_fee_lookup_db,
            )
            if remaining_missing_block_numbers:
                raise SystemExit(
                    f"missing block base-fee enrichment for {len(remaining_missing_block_numbers)} checkpoint blocks "
                    "after checkpoint resume backfill"
                )
        else:
            logging.info(
                "Reusing block base-fee enrichment for checkpoint tx universe from %s",
                block_base_fee_lookup_db.relative_to(repo_root()),
            )
            request_log.append(
                {
                    "source": "eth_getBlockByNumber",
                    "count": 0,
                    "relative_path": str(block_base_fee_lookup_db.relative_to(repo_root())),
                    "fetched_at_utc": None,
                    "reused_existing": True,
                    "reused_via": "block_base_fee_lookup_sqlite_checkpoint",
                }
            )

        rollup_daily, ecosystem_daily, dropped_funding_like_txs = aggregate_daily_from_checkpoint_streaming(
            checkpoint_path=checkpoint_path,
            receipt_lookup_db=receipt_lookup_db,
            block_base_fee_lookup_db=block_base_fee_lookup_db,
        )
    else:
        tx_receipt_hashes = [
            tx_hash for tx_hash, tx in calldata_txs.items() if requires_calldata_receipt_metering(tx)
        ] + list(post_dencun_blob_txs.keys())
        receipt_fields = fetch_receipts(
            snapshot_dir=snapshot_dir,
            tx_hashes=tx_receipt_hashes,
            retries=args.retries,
            timeout_seconds=args.timeout_seconds,
            batch_size=args.rpc_batch_size,
            request_log=request_log,
        )

        for tx_hash, tx in calldata_txs.items():
            receipt = receipt_fields.get(tx_hash)
            if receipt is None:
                continue
            if receipt.block_number != tx.block_number:
                raise SystemExit(f"receipt block mismatch for tx {tx_hash}")

        for tx_hash, blob_tx in post_dencun_blob_txs.items():
            receipt = receipt_fields.get(tx_hash)
            if receipt is None:
                raise SystemExit(f"missing receipt enrichment for Blobscan tx {tx_hash}")
            if receipt.block_number != blob_tx.block_number:
                raise SystemExit(f"receipt block mismatch for tx {tx_hash}")

        block_base_fees = fetch_block_base_fees(
            snapshot_dir=snapshot_dir,
            block_numbers=[tx.block_number for tx in calldata_txs.values()] + [tx.block_number for tx in post_dencun_blob_txs.values()],
            retries=args.retries,
            timeout_seconds=args.timeout_seconds,
            batch_size=args.rpc_batch_size,
            request_log=request_log,
        )

        rollup_daily = {}
        ecosystem_daily = {}
        dropped_funding_like_txs = 0

        def ensure_rollup_bucket(day: str, rollup_id: str) -> dict[str, int]:
            return rollup_daily.setdefault((day, rollup_id), new_rollup_component_bucket())

        def ensure_ecosystem_bucket(day: str) -> dict[str, int]:
            return ecosystem_daily.setdefault(
                day,
                {
                    "base_fee_burn_wei": 0,
                    "blob_fee_burn_wei": 0,
                    "priority_fee_wei": 0,
                    "blob_gas_used": 0,
                    "calldata_gas_proxy": 0,
                },
            )

        for tx in calldata_txs.values():
            receipt = receipt_fields.get(tx.hash)
            gas_used = receipt.gas_used if receipt is not None else tx.gas_used
            if tx.value_wei > 0 and gas_used == 21000:
                dropped_funding_like_txs += 1
                continue
            base_fee_wei = block_base_fees.get(tx.block_number)
            if base_fee_wei is None:
                raise SystemExit(f"missing base fee enrichment for block {tx.block_number}")
            effective_gas_price_wei = receipt.effective_gas_price_wei if receipt is not None else tx.gas_price_wei
            blob_gas_used = receipt.blob_gas_used if receipt is not None else 0
            blob_gas_price_wei = receipt.blob_gas_price_wei if receipt is not None else 0
            priority_fee_wei = gas_used * max(effective_gas_price_wei - base_fee_wei, 0)
            base_burn_wei = gas_used * base_fee_wei
            blob_fee_wei = blob_gas_used * blob_gas_price_wei if blob_gas_used > 0 else 0
            day = tx.timestamp_utc.date().isoformat()

            if not is_canonical_rollup_subtype_in_scope(rollup_id=tx.rollup_id, subtype=tx.subtype):
                continue

            rollup_bucket = ensure_rollup_bucket(day, tx.rollup_id)
            add_rollup_fee_components(
                rollup_bucket,
                subtype=tx.subtype,
                base_burn_wei=base_burn_wei,
                priority_fee_wei=priority_fee_wei,
                blob_fee_wei=blob_fee_wei,
            )

            ecosystem_bucket = ensure_ecosystem_bucket(day)
            ecosystem_bucket["base_fee_burn_wei"] += base_burn_wei
            ecosystem_bucket["priority_fee_wei"] += priority_fee_wei
            ecosystem_bucket["calldata_gas_proxy"] += gas_used
            if blob_gas_used > 0:
                ecosystem_bucket["blob_fee_burn_wei"] += blob_fee_wei
                ecosystem_bucket["blob_gas_used"] += blob_gas_used

        for tx_hash, blob_tx in post_dencun_blob_txs.items():
            receipt = receipt_fields[tx_hash]
            base_fee_wei = block_base_fees.get(blob_tx.block_number)
            if base_fee_wei is None:
                raise SystemExit(f"missing base fee enrichment for block {blob_tx.block_number}")
            priority_fee_wei = receipt.gas_used * max(receipt.effective_gas_price_wei - base_fee_wei, 0)
            base_burn_wei = receipt.gas_used * base_fee_wei
            # Blobscan is the raw source of truth for blob fee fields; legacy receipt caches can
            # legitimately retain zeroed blob fields even when the Blobscan transaction is correct.
            blob_fee_wei = blob_tx.blob_gas_used * blob_tx.blob_gas_price_wei
            day = blob_tx.timestamp_utc.date().isoformat()

            if not is_canonical_rollup_subtype_in_scope(rollup_id=blob_tx.rollup_id, subtype=blob_tx.subtype):
                continue

            rollup_bucket = ensure_rollup_bucket(day, blob_tx.rollup_id)
            add_rollup_fee_components(
                rollup_bucket,
                subtype=blob_tx.subtype,
                base_burn_wei=base_burn_wei,
                priority_fee_wei=priority_fee_wei,
                blob_fee_wei=blob_fee_wei,
            )

            ecosystem_bucket = ensure_ecosystem_bucket(day)
            ecosystem_bucket["base_fee_burn_wei"] += base_burn_wei
            ecosystem_bucket["blob_fee_burn_wei"] += blob_fee_wei
            ecosystem_bucket["priority_fee_wei"] += priority_fee_wei
            ecosystem_bucket["blob_gas_used"] += blob_tx.blob_gas_used
            ecosystem_bucket["calldata_gas_proxy"] += blob_tx.blob_as_calldata_gas_used

    for (day, rollup_id), bucket in sorted(rollup_daily.items()):
        tx_family_total = (
            bucket["batch_submissions_wei"] + bucket["proof_submissions_wei"] + bucket["state_updates_wei"]
        )
        fee_family_total = (
            bucket["execution_base_fee_burn_wei"]
            + bucket["execution_priority_fee_wei"]
            + bucket["blob_fee_burn_wei"]
        )
        if tx_family_total != bucket["rent_paid_wei"]:
            raise SystemExit(
                f"rollup rent component tx-family identity failed for {(day, rollup_id)}: "
                f"{tx_family_total} != {bucket['rent_paid_wei']}"
            )
        if fee_family_total != bucket["rent_paid_wei"]:
            raise SystemExit(
                f"rollup rent component fee-family identity failed for {(day, rollup_id)}: "
                f"{fee_family_total} != {bucket['rent_paid_wei']}"
            )

    panel_rows: list[dict[str, str]] = []
    component_rows: list[dict[str, str]] = []
    vendor_rows.sort(key=lambda row: (row["date_utc"], row["rollup_id"]))
    for row in vendor_rows:
        key = (row["date_utc"], row["rollup_id"])
        onchain = rollup_daily.get(key)
        if onchain is None:
            onchain = new_rollup_component_bucket()
        rent_paid_eth = to_decimal_eth(onchain["rent_paid_wei"])
        panel_rows.append(
            {
                "date_utc": row["date_utc"],
                "rollup_id": row["rollup_id"],
                "l2_fees_eth": row["l2_fees_eth"],
                "rent_paid_eth": format_decimal(rent_paid_eth),
                # Profit is vendor-derived and no longer coherent after replacing vendor rent.
                "profit_eth": "",
                "txcount": row.get("txcount", ""),
            }
        )
        component_rows.append(
            {
                "date_utc": row["date_utc"],
                "rollup_id": row["rollup_id"],
                "batch_submissions_eth": format_decimal(to_decimal_eth(onchain["batch_submissions_wei"])),
                "proof_submissions_eth": format_decimal(to_decimal_eth(onchain["proof_submissions_wei"])),
                "state_updates_eth": format_decimal(to_decimal_eth(onchain["state_updates_wei"])),
                "execution_base_fee_burn_eth": format_decimal(
                    to_decimal_eth(onchain["execution_base_fee_burn_wei"])
                ),
                "execution_priority_fee_eth": format_decimal(
                    to_decimal_eth(onchain["execution_priority_fee_wei"])
                ),
                "blob_fee_burn_eth": format_decimal(to_decimal_eth(onchain["blob_fee_burn_wei"])),
                "rent_paid_eth": format_decimal(rent_paid_eth),
            }
        )

    decomp_rows: list[dict[str, str]] = []
    for day in sorted(ecosystem_daily):
        bucket = ecosystem_daily[day]
        base_burn_wei = coerce_int(str(bucket["base_fee_burn_wei"]))
        blob_burn_wei = coerce_int(str(bucket["blob_fee_burn_wei"]))
        priority_wei = coerce_int(str(bucket["priority_fee_wei"]))
        blob_gas_used = coerce_int(str(bucket["blob_gas_used"]))
        calldata_proxy = coerce_int(str(bucket["calldata_gas_proxy"]))
        total_wei = base_burn_wei + blob_burn_wei + priority_wei
        row = {
            "date_utc": day,
            "l1_base_fee_burn_eth": format_decimal(to_decimal_eth(base_burn_wei)),
            "l1_blob_fee_burn_eth": format_decimal(to_decimal_eth(blob_burn_wei)),
            "l1_priority_fee_eth": format_decimal(to_decimal_eth(priority_wei)),
            "l1_total_rent_eth": format_decimal(to_decimal_eth(total_wei)),
            "l1_blob_gas_used": str(blob_gas_used) if blob_gas_used else "",
            "l1_calldata_gas_used": str(calldata_proxy) if calldata_proxy else "",
            "l1_blob_base_fee_gwei": "",
        }
        if blob_gas_used:
            blob_price_gwei = (Decimal(blob_burn_wei) / Decimal(blob_gas_used)) / WEI_PER_GWEI
            row["l1_blob_base_fee_gwei"] = format_decimal(blob_price_gwei)
        decomp_rows.append(row)

    panel_sample_rows = sample_rows_or_die(panel_rows)
    component_sample_rows = sample_rows_or_die(component_rows)
    decomp_sample_rows = sample_decomp_rows_or_die(decomp_rows)

    write_csv(panel_path, panel_rows, headers=PANEL_HEADERS)
    write_csv(panel_sample_path, panel_sample_rows, headers=PANEL_HEADERS)
    write_csv(component_path, component_rows, headers=COMPONENT_HEADERS)
    write_csv(component_sample_path, component_sample_rows, headers=COMPONENT_HEADERS)
    write_csv(decomp_path, decomp_rows, headers=DECOMP_HEADERS)
    write_csv(decomp_sample_path, decomp_sample_rows, headers=DECOMP_HEADERS)

    fetch_manifest_path = snapshot_dir / "fetch_manifest.json"
    write_json(
        fetch_manifest_path,
        {
            "source": "l1_rent",
            "as_of_utc_date": run_date.isoformat(),
            "command": command_string(args),
            "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
            "protocol_start_utc_date": PROTOCOL_START.isoformat(),
            "observed_end_utc_date": observed_end.isoformat(),
            "dencun_utc_date": DENCUN_DATE.isoformat(),
            "calldata_candidate_txs": calldata_candidate_txs_count,
            "calldata_txs": calldata_txs_count,
            "blob_txs": blob_txs_count,
            "excluded_blob_overlap_txs": excluded_blob_overlap_txs,
            "dropped_funding_like_txs": dropped_funding_like_txs,
            "requests": request_log,
        },
    )

    raw_manifest = build_raw_manifest(
        source="l1_rent",
        snapshot_dir=snapshot_dir,
        command=command_string(args),
        as_of=run_date,
    )
    write_json(raw_manifest_path, raw_manifest)

    decomp_manifest = build_processed_manifest(
        root=root,
        run_date=run_date,
        inputs=[str(raw_manifest_path.relative_to(root))],
        script_path="src/etl/build_l1_rent_panel.py",
        command=command_string(args),
        output_paths=[decomp_path, decomp_sample_path],
    )
    write_json(decomp_manifest_path, decomp_manifest)

    component_manifest = build_processed_manifest(
        root=root,
        run_date=run_date,
        inputs=[
            str(growthepie_raw_manifest_path.relative_to(root)),
            str(raw_manifest_path.relative_to(root)),
            "data/processed/growthepie/vendor_daily_rollup_panel.csv",
        ],
        script_path="src/etl/build_l1_rent_panel.py",
        command=command_string(args),
        output_paths=[component_path, component_sample_path],
    )
    write_json(component_manifest_path, component_manifest)

    panel_manifest = build_processed_manifest(
        root=root,
        run_date=run_date,
        inputs=[
            str(growthepie_raw_manifest_path.relative_to(root)),
            str(raw_manifest_path.relative_to(root)),
            "data/processed/growthepie/vendor_daily_rollup_panel.csv",
        ],
        script_path="src/etl/build_l1_rent_panel.py",
        command=command_string(args),
        output_paths=[panel_path, panel_sample_path],
    )
    write_json(panel_manifest_path, panel_manifest)

    print(f"Wrote raw snapshot to {snapshot_dir.relative_to(root)}")
    print(f"Wrote raw manifest to {raw_manifest_path.relative_to(root)}")
    print(f"Wrote decomposition CSV with {len(decomp_rows)} rows to {decomp_path.relative_to(root)}")
    print(f"Wrote rollup rent component CSV with {len(component_rows)} rows to {component_path.relative_to(root)}")
    print(f"Wrote canonical panel CSV with {len(panel_rows)} rows to {panel_path.relative_to(root)}")
    print(f"Wrote decomposition sample to {decomp_sample_path.relative_to(root)}")
    print(f"Wrote rollup rent component sample to {component_sample_path.relative_to(root)}")
    print(f"Wrote panel sample to {panel_sample_path.relative_to(root)}")
    print(
        "Wrote processed manifests to "
        f"{decomp_manifest_path.relative_to(root)}, "
        f"{component_manifest_path.relative_to(root)}, and "
        f"{panel_manifest_path.relative_to(root)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
