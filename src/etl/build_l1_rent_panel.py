from __future__ import annotations

import argparse
import csv
import hashlib
import http.client
import json
import logging
import platform
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
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
WEI_PER_ETH = Decimal("1000000000000000000")
WEI_PER_GWEI = Decimal("1000000000")

BLOCKSCOUT_TXLIST_URL = "https://eth.blockscout.com/api"
BLOCKSCOUT_RPC_URL = "https://eth.blockscout.com/api/eth-rpc"
BLOBSCAN_TX_URL = "https://api.blobscan.com/transactions"

BLOCKSCOUT_HEADERS = {
    "Accept": "application/json,text/plain,*/*",
    "User-Agent": "Mozilla/5.0",
}
BLOBSCAN_HEADERS = {
    "Accept": "application/json,text/plain,*/*",
    "User-Agent": "Mozilla/5.0",
}

BLOCKSCOUT_TX_PAGE_SIZE = 250
BLOBSCAN_TX_PAGE_SIZE = 500
RPC_BATCH_SIZE = 100

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
SAMPLE_ROLLUPS = ("arbitrum", "base", "optimism")
SAMPLE_DATES = ("2024-03-13", "2024-03-14", "2024-03-15")
ROLLUPS_WITHOUT_BATCHER_ADDRESSES = {"scroll"}


@dataclass(frozen=True)
class RegistryRollup:
    rollup_id: str
    start_date_utc: date
    end_date_utc: date | None
    batcher_addresses: tuple[str, ...]


@dataclass(frozen=True)
class BlockscoutTx:
    hash: str
    rollup_id: str
    address: str
    block_number: int
    timestamp_utc: datetime
    to_address: str | None
    gas_price_wei: int
    gas_used: int
    value_wei: int
    txreceipt_status: str | None


@dataclass(frozen=True)
class BlobscanTx:
    hash: str
    rollup_id: str
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
    return parser.parse_args(argv)


def parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid date {value!r}; expected YYYY-MM-DD") from exc


def parse_datetime(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SystemExit(f"invalid timestamp in source response: {value!r}") from exc


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


def command_string(run_date: date) -> str:
    return " ".join(
        shlex.quote(token)
        for token in ["python", "src/etl/build_l1_rent_panel.py", "--run-date", run_date.isoformat()]
    )


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


def parse_blockscout_tx_record(row: dict[str, Any], *, rollup_id: str, address: str) -> BlockscoutTx:
    raw_to = row.get("to_address")
    to_address = str(raw_to).strip().lower() if isinstance(raw_to, str) and raw_to.strip() else None
    try:
        parsed = BlockscoutTx(
            hash=str(row["hash"]).strip().lower(),
            rollup_id=str(row["rollup_id"]).strip(),
            address=str(row["address"]).strip().lower(),
            block_number=int(row["block_number"]),
            timestamp_utc=parse_datetime(str(row["timestamp_utc"])),
            to_address=to_address,
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
    return parsed


def load_existing_blockscout_page(path: Path, *, rollup_id: str, address: str) -> list[BlockscoutTx]:
    payload = read_json(path)
    rows = payload.get("transactions")
    if not isinstance(rows, list):
        raise SystemExit(f"stored Blockscout page is malformed: {path}")
    return [parse_blockscout_tx_record(row, rollup_id=rollup_id, address=address) for row in rows if isinstance(row, dict)]


def parse_blobscan_tx_record(row: dict[str, Any], *, rollup_id: str) -> BlobscanTx:
    raw_to = row.get("to_address")
    to_address = str(raw_to).strip().lower() if isinstance(raw_to, str) and raw_to.strip() else None
    try:
        parsed = BlobscanTx(
            hash=str(row["hash"]).strip().lower(),
            rollup_id=str(row["rollup_id"]).strip().lower(),
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
    return parsed


def load_existing_blobscan_page(path: Path, *, rollup_id: str) -> tuple[list[BlobscanTx], int | None]:
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
    return [parse_blobscan_tx_record(row, rollup_id=rollup_id) for row in rows if isinstance(row, dict)], total_transactions


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
        except (URLError, TimeoutError, json.JSONDecodeError, http.client.IncompleteRead, http.client.RemoteDisconnected) as exc:
            logging.warning("%s attempt %s/%s failed: %s", url, attempt, retries, exc)

        if attempt < retries:
            time.sleep(delay_seconds)
            delay_seconds *= 2.0

    raise SystemExit(f"source instability or breaking API changes while fetching {url}")


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


def blockscout_window_id(start_dt: datetime, end_dt_exclusive: datetime) -> str:
    if start_dt.timetz() == datetime.min.time().replace(tzinfo=timezone.utc) and end_dt_exclusive.timetz() == datetime.min.time().replace(tzinfo=timezone.utc):
        return window_label(start_dt.date(), end_dt_exclusive.date())

    end_inclusive = end_dt_exclusive - timedelta(seconds=1)
    return (
        f"{start_dt.strftime('%Y-%m-%dT%H%M%SZ')}__"
        f"{end_inclusive.strftime('%Y-%m-%dT%H%M%SZ')}"
    )


def normalize_blockscout_tx(row: dict[str, Any], *, rollup_id: str, address: str) -> BlockscoutTx | None:
    from_address = str(row.get("from", "")).strip().lower()
    if from_address != address:
        return None

    raw_to = row.get("to")
    to_address = str(raw_to).strip().lower() if isinstance(raw_to, str) and raw_to.strip() else None

    try:
        return BlockscoutTx(
            hash=str(row["hash"]).strip().lower(),
            rollup_id=rollup_id,
            address=address,
            block_number=int(str(row["blockNumber"])),
            timestamp_utc=datetime.fromtimestamp(int(str(row["timeStamp"])), tz=timezone.utc),
            to_address=to_address,
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
        "address": tx.address,
        "block_number": tx.block_number,
        "timestamp_utc": tx.timestamp_utc.isoformat(),
        "to_address": tx.to_address,
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
) -> list[BlockscoutTx]:
    window_start_dt = start_timestamp_utc or datetime_utc_start(start_day)
    window_end_exclusive_dt = end_timestamp_exclusive_utc or datetime_utc_start(end_day_exclusive)
    if window_end_exclusive_dt <= window_start_dt:
        return []

    window_id = blockscout_window_id(window_start_dt, window_end_exclusive_dt)
    max_pages = max(1, 10000 // page_size)
    page = 1
    all_rows: list[BlockscoutTx] = []
    window_span_seconds = int((window_end_exclusive_dt - window_start_dt).total_seconds())

    while True:
        page_path = (
            snapshot_dir
            / "blockscout"
            / "txlist"
            / rollup_id
            / address
            / f"{window_id}_page-{page:04d}.json"
        )
        if page_path.exists():
            compact_rows = load_existing_blockscout_page(page_path, rollup_id=rollup_id, address=address)
            request_log.append(
                {
                    "source": "blockscout_txlist",
                    "rollup_id": rollup_id,
                    "address": address,
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
            if len(compact_rows) < page_size:
                break
            page += 1
            continue

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
            "filter_by": "from",
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
            if page == 1 and window_span_seconds > 1:
                split_offset_seconds = max(1, window_span_seconds // 2)
                split_dt = window_start_dt + timedelta(seconds=split_offset_seconds)
                logging.info(
                    "Splitting slow Blockscout tx window for %s/%s from %s..%s into %s..%s and %s..%s",
                    rollup_id,
                    address,
                    window_start_dt.isoformat(),
                    window_end_exclusive_dt.isoformat(),
                    window_start_dt.isoformat(),
                    split_dt.isoformat(),
                    split_dt.isoformat(),
                    window_end_exclusive_dt.isoformat(),
                )
                left_rows = fetch_blockscout_tx_window(
                    snapshot_dir=snapshot_dir,
                    rollup_id=rollup_id,
                    address=address,
                    start_day=window_start_dt.date(),
                    end_day_exclusive=split_dt.date(),
                    page_size=page_size,
                    retries=retries,
                    timeout_seconds=timeout_seconds,
                    request_log=request_log,
                    start_timestamp_utc=window_start_dt,
                    end_timestamp_exclusive_utc=split_dt,
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
                )
                return left_rows + right_rows
            raise
        payload = result.payload
        rows = payload.get("result")
        if not isinstance(rows, list):
            raise SystemExit(f"Blockscout txlist payload is malformed for {rollup_id}/{address}: {payload!r}")

        normalized_rows = [
            normalize_blockscout_tx(row, rollup_id=rollup_id, address=address)
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
                "window_start_utc": window_start_dt.isoformat(),
                "window_end_exclusive_utc": window_end_exclusive_dt.isoformat(),
                "page": page,
                "page_size": page_size,
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


def normalize_blobscan_tx(tx: dict[str, Any], *, default_rollup_id: str | None = None) -> BlobscanTx:
    raw_rollup = tx.get("rollup")
    rollup_id = (
        str(raw_rollup).strip().lower().replace("-", "_")
        if isinstance(raw_rollup, str) and raw_rollup.strip()
        else (default_rollup_id or "")
    )
    if not rollup_id:
        raise SystemExit(f"Blobscan transaction is missing rollup attribution: {tx!r}")

    raw_to = tx.get("to")
    to_address = str(raw_to).strip().lower() if isinstance(raw_to, str) and raw_to.strip() else None

    try:
        return BlobscanTx(
            hash=str(tx["hash"]).strip().lower(),
            rollup_id=rollup_id,
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
        "block_number": tx.block_number,
        "timestamp_utc": tx.timestamp_utc.isoformat(),
        "from_address": tx.from_address,
        "to_address": tx.to_address,
        "blob_gas_used": tx.blob_gas_used,
        "blob_gas_price_wei": str(tx.blob_gas_price_wei),
        "blob_as_calldata_gas_used": tx.blob_as_calldata_gas_used,
    }


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
) -> list[BlobscanTx]:
    if not from_address and not rollup_filter:
        raise ValueError("blobscan fetch requires from_address or rollup_filter")

    page = 1
    total_rows: list[BlobscanTx] = []
    total_transactions: int | None = None
    window_id = window_label(start_day, end_day_exclusive)

    while True:
        source_dir = from_address or f"rollup_{rollup_filter}"
        page_path = snapshot_dir / "blobscan" / rollup_id / source_dir / f"{window_id}_page-{page:04d}.json"
        if page_path.exists():
            normalized_rows, existing_total = load_existing_blobscan_page(page_path, rollup_id=rollup_id)
            if total_transactions is None:
                total_transactions = existing_total
            request_log.append(
                {
                    "source": "blobscan_transactions",
                    "rollup_id": rollup_id,
                    "from_address": from_address,
                    "rollup_filter": rollup_filter,
                    "window_start_utc": start_day.isoformat(),
                    "window_end_exclusive_utc": end_day_exclusive.isoformat(),
                    "page": page,
                    "page_size": page_size,
                    "relative_path": str(page_path.relative_to(repo_root())),
                    "fetched_at_utc": None,
                    "reused_existing": True,
                }
            )
            total_rows.extend(normalized_rows)
            if len(normalized_rows) < page_size:
                break
            if total_transactions is not None and page * page_size >= total_transactions:
                break
            page += 1
            continue

        params: dict[str, str] = {
            "startDate": iso_utc_start(start_day),
            "endDate": iso_utc_end_inclusive(end_day_exclusive),
            "ps": str(page_size),
            "p": str(page),
            "count": "true",
        }
        if from_address:
            params["from"] = from_address
        if rollup_filter:
            params["rollups"] = rollup_filter
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
        if total_transactions is None and payload.get("totalTransactions") is not None:
            try:
                total_transactions = int(payload["totalTransactions"])
            except (TypeError, ValueError) as exc:
                raise SystemExit(f"Blobscan totalTransactions is malformed for {rollup_id}: {payload!r}") from exc

        normalized_rows = [
            normalize_blobscan_tx(row, default_rollup_id=rollup_id)
            for row in rows
            if isinstance(row, dict)
        ]
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
                "window_start_utc": start_day.isoformat(),
                "window_end_exclusive_utc": end_day_exclusive.isoformat(),
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
                "window_start_utc": start_day.isoformat(),
                "window_end_exclusive_utc": end_day_exclusive.isoformat(),
                "page": page,
                "page_size": page_size,
                "relative_path": str(page_path.relative_to(repo_root())),
                "fetched_at_utc": result.fetched_at_utc,
            }
        )
        total_rows.extend(normalized_rows)
        if len(rows) < page_size:
            break
        if total_transactions is not None and page * page_size >= total_transactions:
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


def fetch_block_base_fees(
    *,
    snapshot_dir: Path,
    block_numbers: Iterable[int],
    retries: int,
    timeout_seconds: float,
    batch_size: int,
    request_log: list[dict[str, Any]],
) -> dict[int, int]:
    known: dict[int, int] = {}
    batch_index = 1
    for block_chunk in chunked(sorted(set(block_numbers)), batch_size):
        path = snapshot_dir / "blockscout" / "block_base_fees" / f"batch-{batch_index:04d}.json"
        if path.exists():
            known.update(load_existing_block_fee_batch(path))
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
            batch_index += 1
            continue

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
        result = fetch_json(
            BLOCKSCOUT_RPC_URL,
            headers={"Content-Type": "application/json", **BLOCKSCOUT_HEADERS},
            retries=retries,
            timeout_seconds=timeout_seconds,
            method="POST",
            body=body,
        )
        payload = result.payload
        if not isinstance(payload, list):
            raise SystemExit(f"eth_getBlockByNumber batch response is malformed: {payload!r}")

        batch_rows: list[dict[str, Any]] = []
        for item in payload:
            if not isinstance(item, dict):
                raise SystemExit(f"eth_getBlockByNumber batch item is malformed: {item!r}")
            block = item.get("result")
            if not isinstance(block, dict):
                raise SystemExit(f"eth_getBlockByNumber returned malformed result: {item!r}")
            try:
                block_number = int(str(block["number"]), 16)
                base_fee_wei = int(str(block.get("baseFeePerGas", "0x0")), 16)
                timestamp_utc = datetime.fromtimestamp(int(str(block["timestamp"]), 16), tz=timezone.utc).isoformat()
            except (KeyError, TypeError, ValueError) as exc:
                raise SystemExit(f"eth_getBlockByNumber result is malformed: {block!r}") from exc

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
                "fetched_at_utc": result.fetched_at_utc,
                "blocks": batch_rows,
            },
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
) -> dict[str, ReceiptFields]:
    receipts: dict[str, ReceiptFields] = {}
    batch_index = 1
    for hash_chunk in chunked(sorted(set(tx_hashes)), batch_size):
        path = snapshot_dir / "blockscout" / "receipts" / f"batch-{batch_index:04d}.json"
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

        requests = [
            {
                "id": index,
                "jsonrpc": "2.0",
                "method": "eth_getTransactionReceipt",
                "params": [tx_hash],
            }
            for index, tx_hash in enumerate(hash_chunk, start=1)
        ]
        body = json.dumps(requests).encode("utf-8")
        result = fetch_json(
            BLOCKSCOUT_RPC_URL,
            headers={"Content-Type": "application/json", **BLOCKSCOUT_HEADERS},
            retries=retries,
            timeout_seconds=timeout_seconds,
            method="POST",
            body=body,
        )
        payload = result.payload
        if not isinstance(payload, list):
            raise SystemExit(f"eth_getTransactionReceipt batch response is malformed: {payload!r}")

        compact_rows: list[dict[str, Any]] = []
        for item in payload:
            if not isinstance(item, dict):
                raise SystemExit(f"eth_getTransactionReceipt batch item is malformed: {item!r}")
            receipt = item.get("result")
            if not isinstance(receipt, dict):
                raise SystemExit(f"eth_getTransactionReceipt returned malformed result: {item!r}")
            try:
                tx_hash = str(receipt["transactionHash"]).strip().lower()
                raw_to = receipt.get("to")
                to_address = str(raw_to).strip().lower() if isinstance(raw_to, str) and raw_to.strip() else None
                row = ReceiptFields(
                    hash=tx_hash,
                    block_number=int(str(receipt["blockNumber"]), 16),
                    from_address=str(receipt["from"]).strip().lower(),
                    to_address=to_address,
                    gas_used=int(str(receipt["gasUsed"]), 16),
                    effective_gas_price_wei=int(str(receipt["effectiveGasPrice"]), 16),
                    blob_gas_used=int(str(receipt.get("blobGasUsed", "0x0")), 16),
                    blob_gas_price_wei=int(str(receipt.get("blobGasPrice", "0x0")), 16),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise SystemExit(f"eth_getTransactionReceipt result is malformed: {receipt!r}") from exc

            receipts[row.hash] = row
            compact_rows.append(
                {
                    "hash": row.hash,
                    "block_number": row.block_number,
                    "from_address": row.from_address,
                    "to_address": row.to_address,
                    "gas_used": row.gas_used,
                    "effective_gas_price_wei": str(row.effective_gas_price_wei),
                    "blob_gas_used": row.blob_gas_used,
                    "blob_gas_price_wei": str(row.blob_gas_price_wei),
                }
            )

        write_json(
            path,
            {
                "source": "eth_getTransactionReceipt",
                "fetched_at_utc": result.fetched_at_utc,
                "receipts": compact_rows,
            },
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
    files: list[dict[str, Any]] = []
    for path in sorted(snapshot_dir.rglob("*")):
        if not path.is_file():
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
    output_paths: list[Path],
) -> dict[str, Any]:
    return {
        "as_of_utc_date": run_date.isoformat(),
        "inputs": inputs,
        "transform": {
            "script_path": script_path,
            "git_sha": git_sha(root),
            "command": command_string(run_date),
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


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    root = repo_root()
    run_date = args.run_date
    observed_end = observed_end_date(run_date)
    if observed_end < PROTOCOL_START:
        raise SystemExit(f"run-date {run_date.isoformat()} is before the protocol start window")

    snapshot_dir = root / "data" / "raw" / "l1_rent" / run_date.isoformat()
    raw_manifest_path = root / "data" / "raw_manifest" / f"l1_rent_{run_date.isoformat()}.json"
    decomp_path = root / "data" / "processed" / "l1_rent" / "daily_l1_rent_decomposition.csv"
    panel_path = root / "data" / "processed" / "panels" / "daily_rollup_panel.csv"
    decomp_sample_path = root / "data" / "samples" / "l1_rent" / "daily_l1_rent_decomposition_sample.csv"
    panel_sample_path = root / "data" / "samples" / "panels" / "daily_rollup_panel_sample.csv"
    decomp_manifest_path = (
        root / "data" / "processed_manifest" / f"daily_l1_rent_decomposition_{run_date.isoformat()}.json"
    )
    panel_manifest_path = root / "data" / "processed_manifest" / f"daily_rollup_panel_{run_date.isoformat()}.json"

    prepare_snapshot_dir(snapshot_dir, raw_manifest_path=raw_manifest_path)
    ensure_new_manifest(raw_manifest_path, label="raw manifest")
    ensure_new_manifest(decomp_manifest_path, label="processed decomposition manifest")
    ensure_new_manifest(panel_manifest_path, label="processed panel manifest")

    registry_path = root / "registry" / "rollup_registry_v1.csv"
    growthepie_raw_manifest_path = root / "data" / "raw_manifest" / f"growthepie_{run_date.isoformat()}.json"
    if not growthepie_raw_manifest_path.exists():
        raise SystemExit(f"required growthepie raw manifest is missing: {growthepie_raw_manifest_path}")
    vendor_panel_path = root / "data" / "processed" / "growthepie" / "vendor_daily_rollup_panel.csv"

    rollups = load_registry(registry_path)
    vendor_rows = load_vendor_panel(vendor_panel_path)

    request_log: list[dict[str, Any]] = []
    pre_dencun_txs: dict[str, BlockscoutTx] = {}
    post_dencun_blob_txs: dict[str, BlobscanTx] = {}

    for rollup in rollups:
        active_start = max(PROTOCOL_START, rollup.start_date_utc)
        active_end = observed_end if rollup.end_date_utc is None else min(observed_end, rollup.end_date_utc)
        if active_end < active_start:
            continue

        if rollup.batcher_addresses:
            pre_end = min(active_end, DENCUN_DATE - timedelta(days=1))
            if pre_end >= active_start:
                for address in rollup.batcher_addresses:
                    for window_start, window_end_exclusive in month_windows(active_start, pre_end):
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
                        )
                        for row in rows:
                            if row.hash in pre_dencun_txs:
                                raise SystemExit(
                                    f"on-chain attribution is ambiguous: duplicate pre-Dencun tx hash {row.hash} "
                                    f"for {rollup.rollup_id}"
                                )
                            pre_dencun_txs[row.hash] = row

        post_start = max(active_start, DENCUN_DATE)
        if active_end < post_start:
            continue
        for window_start, window_end_exclusive in month_windows(post_start, active_end):
            if rollup.batcher_addresses:
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

    receipt_fields = fetch_receipts(
        snapshot_dir=snapshot_dir,
        tx_hashes=post_dencun_blob_txs.keys(),
        retries=args.retries,
        timeout_seconds=args.timeout_seconds,
        batch_size=args.rpc_batch_size,
        request_log=request_log,
    )

    for tx_hash, blob_tx in post_dencun_blob_txs.items():
        receipt = receipt_fields.get(tx_hash)
        if receipt is None:
            raise SystemExit(f"missing receipt enrichment for Blobscan tx {tx_hash}")
        if receipt.block_number != blob_tx.block_number:
            raise SystemExit(f"receipt block mismatch for tx {tx_hash}")

    block_base_fees = fetch_block_base_fees(
        snapshot_dir=snapshot_dir,
        block_numbers=[tx.block_number for tx in pre_dencun_txs.values()]
        + [tx.block_number for tx in post_dencun_blob_txs.values()],
        retries=args.retries,
        timeout_seconds=args.timeout_seconds,
        batch_size=args.rpc_batch_size,
        request_log=request_log,
    )

    # The post-Dencun slice comes from Blobscan + receipts; the calldata-era slice comes
    # from Blockscout address history. This keeps the decomposition aligned to the
    # registry-backed switch from calldata to blobs at Dencun.
    rollup_daily: dict[tuple[str, str], dict[str, Decimal | int]] = {}
    ecosystem_daily: dict[str, dict[str, Decimal | int]] = {}
    dropped_funding_like_txs = 0

    def ensure_rollup_bucket(day: str, rollup_id: str) -> dict[str, Decimal | int]:
        return rollup_daily.setdefault(
            (day, rollup_id),
            {
                "rent_paid_wei": 0,
            },
        )

    def ensure_ecosystem_bucket(day: str) -> dict[str, Decimal | int]:
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

    for tx in pre_dencun_txs.values():
        if tx.value_wei > 0 and tx.gas_used == 21000:
            dropped_funding_like_txs += 1
            continue
        base_fee_wei = block_base_fees.get(tx.block_number)
        if base_fee_wei is None:
            raise SystemExit(f"missing base fee enrichment for block {tx.block_number}")
        effective_gas_price_wei = tx.gas_price_wei
        priority_fee_wei = tx.gas_used * max(effective_gas_price_wei - base_fee_wei, 0)
        base_burn_wei = tx.gas_used * base_fee_wei
        total_rent_wei = base_burn_wei + priority_fee_wei
        day = tx.timestamp_utc.date().isoformat()

        rollup_bucket = ensure_rollup_bucket(day, tx.rollup_id)
        rollup_bucket["rent_paid_wei"] = coerce_int(str(rollup_bucket["rent_paid_wei"])) + total_rent_wei

        ecosystem_bucket = ensure_ecosystem_bucket(day)
        ecosystem_bucket["base_fee_burn_wei"] = coerce_int(str(ecosystem_bucket["base_fee_burn_wei"])) + base_burn_wei
        ecosystem_bucket["priority_fee_wei"] = coerce_int(str(ecosystem_bucket["priority_fee_wei"])) + priority_fee_wei
        ecosystem_bucket["calldata_gas_proxy"] = coerce_int(str(ecosystem_bucket["calldata_gas_proxy"])) + tx.gas_used

    for tx_hash, blob_tx in post_dencun_blob_txs.items():
        receipt = receipt_fields[tx_hash]
        base_fee_wei = block_base_fees.get(blob_tx.block_number)
        if base_fee_wei is None:
            raise SystemExit(f"missing base fee enrichment for block {blob_tx.block_number}")
        priority_fee_wei = receipt.gas_used * max(receipt.effective_gas_price_wei - base_fee_wei, 0)
        base_burn_wei = receipt.gas_used * base_fee_wei
        blob_fee_wei = receipt.blob_gas_used * receipt.blob_gas_price_wei
        total_rent_wei = base_burn_wei + priority_fee_wei + blob_fee_wei
        day = blob_tx.timestamp_utc.date().isoformat()

        rollup_bucket = ensure_rollup_bucket(day, blob_tx.rollup_id)
        rollup_bucket["rent_paid_wei"] = coerce_int(str(rollup_bucket["rent_paid_wei"])) + total_rent_wei

        ecosystem_bucket = ensure_ecosystem_bucket(day)
        ecosystem_bucket["base_fee_burn_wei"] = coerce_int(str(ecosystem_bucket["base_fee_burn_wei"])) + base_burn_wei
        ecosystem_bucket["blob_fee_burn_wei"] = coerce_int(str(ecosystem_bucket["blob_fee_burn_wei"])) + blob_fee_wei
        ecosystem_bucket["priority_fee_wei"] = coerce_int(str(ecosystem_bucket["priority_fee_wei"])) + priority_fee_wei
        ecosystem_bucket["blob_gas_used"] = coerce_int(str(ecosystem_bucket["blob_gas_used"])) + receipt.blob_gas_used
        ecosystem_bucket["calldata_gas_proxy"] = (
            coerce_int(str(ecosystem_bucket["calldata_gas_proxy"])) + blob_tx.blob_as_calldata_gas_used
        )

    panel_rows: list[dict[str, str]] = []
    vendor_rows.sort(key=lambda row: (row["date_utc"], row["rollup_id"]))
    for row in vendor_rows:
        key = (row["date_utc"], row["rollup_id"])
        onchain = rollup_daily.get(key)
        if onchain is None:
            continue
        rent_paid_eth = to_decimal_eth(coerce_int(str(onchain["rent_paid_wei"])))
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
    decomp_sample_rows = sample_decomp_rows_or_die(decomp_rows)

    write_csv(panel_path, panel_rows, headers=PANEL_HEADERS)
    write_csv(panel_sample_path, panel_sample_rows, headers=PANEL_HEADERS)
    write_csv(decomp_path, decomp_rows, headers=DECOMP_HEADERS)
    write_csv(decomp_sample_path, decomp_sample_rows, headers=DECOMP_HEADERS)

    fetch_manifest_path = snapshot_dir / "fetch_manifest.json"
    write_json(
        fetch_manifest_path,
        {
            "source": "l1_rent",
            "as_of_utc_date": run_date.isoformat(),
            "command": command_string(run_date),
            "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
            "protocol_start_utc_date": PROTOCOL_START.isoformat(),
            "observed_end_utc_date": observed_end.isoformat(),
            "dencun_utc_date": DENCUN_DATE.isoformat(),
            "dropped_funding_like_txs": dropped_funding_like_txs,
            "requests": request_log,
        },
    )

    raw_manifest = build_raw_manifest(
        source="l1_rent",
        snapshot_dir=snapshot_dir,
        command=command_string(run_date),
        as_of=run_date,
    )
    write_json(raw_manifest_path, raw_manifest)

    decomp_manifest = build_processed_manifest(
        root=root,
        run_date=run_date,
        inputs=[str(raw_manifest_path.relative_to(root))],
        script_path="src/etl/build_l1_rent_panel.py",
        output_paths=[decomp_path, decomp_sample_path],
    )
    write_json(decomp_manifest_path, decomp_manifest)

    panel_manifest = build_processed_manifest(
        root=root,
        run_date=run_date,
        inputs=[
            str(growthepie_raw_manifest_path.relative_to(root)),
            str(raw_manifest_path.relative_to(root)),
            "data/processed/growthepie/vendor_daily_rollup_panel.csv",
        ],
        script_path="src/etl/build_l1_rent_panel.py",
        output_paths=[panel_path, panel_sample_path],
    )
    write_json(panel_manifest_path, panel_manifest)

    print(f"Wrote raw snapshot to {snapshot_dir.relative_to(root)}")
    print(f"Wrote raw manifest to {raw_manifest_path.relative_to(root)}")
    print(f"Wrote decomposition CSV with {len(decomp_rows)} rows to {decomp_path.relative_to(root)}")
    print(f"Wrote canonical panel CSV with {len(panel_rows)} rows to {panel_path.relative_to(root)}")
    print(f"Wrote decomposition sample to {decomp_sample_path.relative_to(root)}")
    print(f"Wrote panel sample to {panel_sample_path.relative_to(root)}")
    print(f"Wrote processed manifests to {decomp_manifest_path.relative_to(root)} and {panel_manifest_path.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
