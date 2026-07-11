from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import logging
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from pack_config import dataframe_schema_field_names, load_pack_config, pack_value  # noqa: E402
from manifest_tools import build_processed_manifest as build_executable_processed_manifest  # noqa: E402


MASTER_URL = "https://api.growthepie.com/v1/master.json"
CHAIN_METRIC_URL = "https://api.growthepie.com/v1/metrics/chains/{chain}/{metric}.json"
PROTOCOL_START = date(2022, 1, 1)
OUTPUT_HEADERS = list(
    dataframe_schema_field_names(Path(__file__).resolve().parents[2], "paths.primary_panel_schema")
)
SAMPLE_ROLLUPS = ("arbitrum", "base", "optimism")
SAMPLE_DATES = ("2024-03-13", "2024-03-14", "2024-03-15")
REQUIRED_METRICS = ("fees", "rent_paid")
OPTIONAL_METRICS = ("profit", "txcount")
ALL_METRICS = REQUIRED_METRICS + OPTIONAL_METRICS
PROFIT_ABS_TOLERANCE = Decimal("1e-9")
PROFIT_REL_TOLERANCE = Decimal("0.01")
HTTP_HEADERS = {
    "Accept": "application/json,text/plain,*/*",
    "Origin": "https://www.growthepie.com",
    "Referer": "https://www.growthepie.com/",
    "User-Agent": "Mozilla/5.0",
}
OPTIONAL_MISSING_STATUS_CODES = {403, 404}


@dataclass(frozen=True)
class RegistryRollup:
    rollup_id: str
    start_date_utc: date
    end_date_utc: date | None


@dataclass(frozen=True)
class FetchResult:
    payload: Any
    raw_bytes: bytes
    fetched_at_utc: str


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="growthepie_fetch.py")
    parser.add_argument("--run-date", required=True, type=parse_date, help="UTC as-of date (YYYY-MM-DD)")
    parser.add_argument("--retries", type=int, default=4, help="Retry attempts for network fetches")
    parser.add_argument("--timeout-seconds", type=float, default=30.0, help="Per-request timeout in seconds")
    return parser.parse_args(argv)


def parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid date {value!r}; expected YYYY-MM-DD") from exc


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


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
        for token in ["python", "src/etl/growthepie_fetch.py", "--run-date", run_date.isoformat()]
    )


def ensure_fresh_snapshot_dir(path: Path) -> None:
    if path.exists():
        try:
            next(path.iterdir())
        except StopIteration:
            return
        raise SystemExit(
            f"append-only raw snapshot already exists and is non-empty: {path}. "
            "Choose a new --run-date instead of overwriting prior pulls."
        )
    path.mkdir(parents=True, exist_ok=False)


def ensure_new_processed_manifest(path: Path) -> None:
    if path.exists():
        raise SystemExit(
            f"processed manifest already exists for this run date: {path}. "
            "Manifests are append-only; do not overwrite prior provenance."
        )


def fetch_json(url: str, *, retries: int, timeout_seconds: float, optional: bool = False) -> FetchResult | None:
    delay_seconds = 1.0
    for attempt in range(1, retries + 1):
        request = Request(url, headers=HTTP_HEADERS)
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                raw_bytes = response.read()
            fetched_at = datetime.now(timezone.utc).isoformat()
            payload = json.loads(raw_bytes.decode("utf-8"), parse_float=Decimal)
            return FetchResult(payload=payload, raw_bytes=raw_bytes, fetched_at_utc=fetched_at)
        except HTTPError as exc:
            body = exc.read(200).decode("utf-8", errors="replace")
            message = f"{url} attempt {attempt}/{retries} failed with HTTP {exc.code}: {body.strip()!r}"
            if optional and exc.code in OPTIONAL_MISSING_STATUS_CODES:
                logging.warning("%s", message)
                return None
            logging.warning("%s", message)
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            logging.warning("%s attempt %s/%s failed: %s", url, attempt, retries, exc)

        if attempt < retries:
            time.sleep(delay_seconds)
            delay_seconds *= 2.0

    raise SystemExit(f"source instability or breaking API changes while fetching {url}")


def write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def load_registry(path: Path) -> list[RegistryRollup]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    rollups: list[RegistryRollup] = []
    for row in rows:
        rollups.append(
            RegistryRollup(
                rollup_id=row["rollup_id"],
                start_date_utc=parse_date(row["start_date_utc"]),
                end_date_utc=parse_date(row["end_date_utc"]) if row["end_date_utc"] else None,
            )
        )
    return rollups


def validate_registry_against_master(master_payload: dict[str, Any], rollups: list[RegistryRollup]) -> None:
    chains = master_payload.get("chains")
    if not isinstance(chains, dict):
        raise SystemExit("master.json is missing the chains section")

    for rollup in rollups:
        chain = chains.get(rollup.rollup_id)
        if not isinstance(chain, dict):
            raise SystemExit(
                f"registry identifier required for normalization is missing from master.json: {rollup.rollup_id}"
            )
        supported_metrics = chain.get("supported_metrics")
        if not isinstance(supported_metrics, list):
            raise SystemExit(f"master.json missing supported_metrics for {rollup.rollup_id}")
        missing_required = [metric for metric in REQUIRED_METRICS if metric not in supported_metrics]
        if missing_required:
            raise SystemExit(
                f"source instability for {rollup.rollup_id}: missing required metrics {missing_required}"
            )


def build_metric_index(payload: dict[str, Any], *, metric: str) -> dict[date, Decimal | int]:
    details = payload.get("details")
    if not isinstance(details, dict):
        raise SystemExit(f"{metric} payload missing details object")
    timeseries = details.get("timeseries")
    if not isinstance(timeseries, dict):
        raise SystemExit(f"{metric} payload missing timeseries object")
    daily = timeseries.get("daily")
    if not isinstance(daily, dict):
        raise SystemExit(f"{metric} payload missing daily timeseries")
    types = daily.get("types")
    rows = daily.get("data")
    if not isinstance(types, list) or not isinstance(rows, list):
        raise SystemExit(f"{metric} payload daily timeseries is malformed")

    index_by_name = {name: idx for idx, name in enumerate(types)}
    value_key = "value" if metric == "txcount" else "eth"
    if "unix" not in index_by_name or value_key not in index_by_name:
        raise SystemExit(f"{metric} payload daily types missing unix/{value_key}: {types}")

    unix_idx = index_by_name["unix"]
    value_idx = index_by_name[value_key]

    out: dict[date, Decimal | int] = {}
    for row in rows:
        if not isinstance(row, list):
            raise SystemExit(f"{metric} payload row is not a list: {row!r}")
        if max(unix_idx, value_idx) >= len(row):
            raise SystemExit(f"{metric} payload row is shorter than declared types: {row!r}")

        raw_unix = row[unix_idx]
        raw_value = row[value_idx]
        if raw_unix is None or raw_value is None:
            continue
        if not isinstance(raw_unix, (int, float, Decimal)):
            raise SystemExit(f"{metric} unix timestamp is not numeric: {raw_unix!r}")

        timestamp = datetime.fromtimestamp(float(raw_unix) / 1000.0, tz=timezone.utc).date()
        if metric == "txcount":
            decimal_value = raw_value if isinstance(raw_value, Decimal) else Decimal(str(raw_value))
            integral_value = decimal_value.to_integral_value()
            if decimal_value != integral_value:
                raise SystemExit(f"txcount value is not integer-like on {timestamp.isoformat()}: {raw_value!r}")
            out[timestamp] = int(integral_value)
        else:
            out[timestamp] = raw_value if isinstance(raw_value, Decimal) else Decimal(str(raw_value))

    return out


def format_decimal(value: Decimal) -> str:
    text = format(value.normalize(), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text if text not in {"", "-0"} else "0"


def require_decimal_metric(value: Decimal | int, *, metric: str) -> Decimal:
    if isinstance(value, Decimal):
        return value
    raise SystemExit(f"{metric} normalization expected Decimal values, got {value!r}")


def profit_within_protocol_tolerance(*, fees_value: Decimal, rent_value: Decimal, profit_value: Decimal) -> bool:
    implied_profit = fees_value - rent_value
    tolerance = max(
        PROFIT_ABS_TOLERANCE,
        PROFIT_REL_TOLERANCE * max(abs(fees_value), abs(rent_value), PROFIT_ABS_TOLERANCE),
    )
    return abs(profit_value - implied_profit) <= tolerance


def row_with_optional_values(
    *,
    day: date,
    rollup_id: str,
    fees_by_day: dict[date, Decimal | int],
    rent_by_day: dict[date, Decimal | int],
    profit_by_day: dict[date, Decimal | int] | None,
    txcount_by_day: dict[date, Decimal | int] | None,
) -> tuple[dict[str, str], bool]:
    fees_value = require_decimal_metric(fees_by_day[day], metric="fees")
    rent_value = require_decimal_metric(rent_by_day[day], metric="rent_paid")
    omitted_profit = False
    row = {
        "date_utc": day.isoformat(),
        "rollup_id": rollup_id,
        "l2_fees_eth": format_decimal(fees_value),
        "rent_paid_eth": format_decimal(rent_value),
        "profit_eth": "",
        "txcount": "",
    }
    if profit_by_day and day in profit_by_day:
        # Only surface vendor profit when the reported series satisfies the locked
        # accounting identity against the same vendor fees and rent inputs.
        profit_value = require_decimal_metric(profit_by_day[day], metric="profit")
        if profit_within_protocol_tolerance(
            fees_value=fees_value,
            rent_value=rent_value,
            profit_value=profit_value,
        ):
            row["profit_eth"] = format_decimal(profit_value)
        else:
            omitted_profit = True
    if txcount_by_day and day in txcount_by_day:
        row["txcount"] = str(txcount_by_day[day])
    return row, omitted_profit


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_HEADERS)
        writer.writeheader()
        writer.writerows(rows)


def build_processed_manifest(
    *,
    root: Path,
    run_date: date,
    output_paths: list[Path],
) -> dict[str, Any]:
    return build_executable_processed_manifest(
        repo=root,
        as_of_utc_date=run_date.isoformat(),
        inputs=[f"data/raw_manifest/growthepie_{run_date.isoformat()}.json"],
        script_path="src/etl/growthepie_fetch.py",
        command=command_string(run_date),
        outputs=output_paths,
        allow_dirty_with_diff=True,
    )


def sample_rows_or_die(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_key = {(row["date_utc"], row["rollup_id"]): row for row in rows}
    sample: list[dict[str, str]] = []
    # Fixed historical rows keep the tracked sample stable across reruns.
    for day in SAMPLE_DATES:
        for rollup_id in SAMPLE_ROLLUPS:
            key = (day, rollup_id)
            if key not in by_key:
                raise SystemExit(f"sample selection missing required panel row: {key}")
            sample.append(by_key[key])
    return sample


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    root = repo_root()
    pack = load_pack_config(root)
    run_date = args.run_date
    snapshot_dir = root / "data" / "raw" / "growthepie" / run_date.isoformat()
    processed_dir = root / "data" / "processed" / "growthepie"
    panel_path = root / pack_value(pack, "paths.vendor_panel")
    sample_path = root / pack_value(pack, "paths.vendor_panel_sample")
    processed_manifest_path = root / pack_value(
        pack,
        "paths.vendor_panel_manifest_pattern",
    ).format(date=run_date.isoformat())
    registry_path = root / pack_value(pack, "paths.registry")

    ensure_fresh_snapshot_dir(snapshot_dir)
    ensure_new_processed_manifest(processed_manifest_path)

    master_result = fetch_json(
        MASTER_URL,
        retries=args.retries,
        timeout_seconds=args.timeout_seconds,
    )
    assert master_result is not None
    write_bytes(snapshot_dir / "master.json", master_result.raw_bytes)

    rollups = load_registry(registry_path)
    validate_registry_against_master(master_result.payload, rollups)

    fetch_manifest_requests: list[dict[str, Any]] = [
        {
            "url": MASTER_URL,
            "relative_path": str((snapshot_dir / "master.json").relative_to(root)),
            "fetched_at_utc": master_result.fetched_at_utc,
            "last_updated_utc": master_result.payload.get("last_updated_utc"),
        }
    ]

    all_rows: list[dict[str, str]] = []
    omitted_profit_rows = 0
    omitted_profit_by_rollup: Counter[str] = Counter()
    master_chains = master_result.payload["chains"]
    for rollup in rollups:
        chain_meta = master_chains[rollup.rollup_id]
        supported_metrics = set(chain_meta.get("supported_metrics", []))
        series_by_metric: dict[str, dict[date, Decimal | int]] = {}

        for metric in ALL_METRICS:
            if metric in OPTIONAL_METRICS and metric not in supported_metrics:
                logging.info("Skipping unsupported optional metric %s for %s", metric, rollup.rollup_id)
                continue

            metric_url = CHAIN_METRIC_URL.format(chain=rollup.rollup_id, metric=metric)
            result = fetch_json(
                metric_url,
                retries=args.retries,
                timeout_seconds=args.timeout_seconds,
                optional=metric in OPTIONAL_METRICS,
            )
            if result is None:
                fetch_manifest_requests.append(
                    {
                        "url": metric_url,
                        "relative_path": None,
                        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
                        "rollup_id": rollup.rollup_id,
                        "metric": metric,
                        "status": "optional_missing",
                    }
                )
                continue

            raw_path = snapshot_dir / "metrics" / "chains" / rollup.rollup_id / f"{metric}.json"
            write_bytes(raw_path, result.raw_bytes)
            series_by_metric[metric] = build_metric_index(result.payload, metric=metric)
            fetch_manifest_requests.append(
                {
                    "url": metric_url,
                    "relative_path": str(raw_path.relative_to(root)),
                    "fetched_at_utc": result.fetched_at_utc,
                    "rollup_id": rollup.rollup_id,
                    "metric": metric,
                    "last_updated_utc": result.payload.get("last_updated_utc"),
                    "status": "ok",
                }
            )

        fees_by_day = series_by_metric["fees"]
        rent_by_day = series_by_metric["rent_paid"]
        profit_by_day = series_by_metric.get("profit")
        txcount_by_day = series_by_metric.get("txcount")

        active_start = max(PROTOCOL_START, rollup.start_date_utc)
        active_end = min(run_date, rollup.end_date_utc) if rollup.end_date_utc else run_date
        if active_end < active_start:
            continue

        eligible_days = sorted(
            day
            for day in (set(fees_by_day) & set(rent_by_day))
            if active_start <= day <= active_end
        )
        for day in eligible_days:
            row, omitted_profit = row_with_optional_values(
                day=day,
                rollup_id=rollup.rollup_id,
                fees_by_day=fees_by_day,
                rent_by_day=rent_by_day,
                profit_by_day=profit_by_day,
                txcount_by_day=txcount_by_day,
            )
            all_rows.append(row)
            if omitted_profit:
                omitted_profit_rows += 1
                omitted_profit_by_rollup[rollup.rollup_id] += 1

    all_rows.sort(key=lambda row: (row["date_utc"], row["rollup_id"]))
    sample_rows = sample_rows_or_die(all_rows)

    if omitted_profit_rows:
        logging.info(
            "Omitted %s vendor profit rows that failed the protocol accounting identity: %s",
            omitted_profit_rows,
            dict(sorted(omitted_profit_by_rollup.items())),
        )

    write_csv(panel_path, all_rows)
    write_csv(sample_path, sample_rows)

    fetch_manifest = {
        "source": "growthepie",
        "as_of_utc_date": run_date.isoformat(),
        "command": command_string(run_date),
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "rollups": [rollup.rollup_id for rollup in rollups],
        "metrics": list(ALL_METRICS),
        "requests": fetch_manifest_requests,
    }
    with (snapshot_dir / "fetch_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(fetch_manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    processed_manifest = build_processed_manifest(
        root=root,
        run_date=run_date,
        output_paths=[panel_path, sample_path],
    )
    processed_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with processed_manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(processed_manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"Wrote raw snapshot to {snapshot_dir.relative_to(root)}")
    print(f"Wrote panel CSV with {len(all_rows)} rows to {panel_path.relative_to(root)}")
    print(f"Wrote sample CSV with {len(sample_rows)} rows to {sample_path.relative_to(root)}")
    print(f"Wrote processed manifest to {processed_manifest_path.relative_to(root)}")
    print(
        "Next step: python scripts/make_raw_manifest.py growthepie "
        f"data/raw/growthepie/{run_date.isoformat()} --as-of {run_date.isoformat()} -- "
        f"{command_string(run_date)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
