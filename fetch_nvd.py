"""
Simple script to download CVE entries from the NVD API and persist them as CSV.

Usage example:

    python fetch_nvd.py --start-year 2020 --end-year 2020 --output cves_2020.csv
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import logging
import random
import time
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import requests


API_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
ALLOWED_SEVERITIES = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
DEFAULT_DELAY = 0.6  # seconds – respects NVD unauthenticated rate limits
MAX_RESULTS_PER_REQUEST = 2000

LOGGER = logging.getLogger("fetch_nvd")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch CVE data from NVD and save to CSV.")
    parser.add_argument("--start-year", type=int, required=True, help="First year to download (inclusive).")
    parser.add_argument("--end-year", type=int, required=True, help="Last year to download (inclusive).")
    parser.add_argument("--output", type=Path, required=True, help="Destination CSV file.")
    parser.add_argument("--api-key", type=str, help="Optional NVD API key for higher rate limits.")
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY,
        help="Delay between requests (seconds). Increase if rate-limited.",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=30,
        help="Window size (in days) for each request. Smaller windows reduce pagination.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional number of CVEs to sample before writing the CSV.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed used when sampling.",
    )
    
    return parser.parse_args()


def daterange(start: dt.datetime, end: dt.datetime, step_days: int) -> Iterator[Tuple[dt.datetime, dt.datetime]]:
    cursor = start
    delta = dt.timedelta(days=step_days)
    while cursor < end:
        next_cursor = min(cursor + delta, end)
        yield cursor, next_cursor
        cursor = next_cursor


def build_session(api_key: Optional[str]) -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": "bayesstrike-fetch-nvd"})
    if api_key:
        session.headers["apiKey"] = api_key
    return session


def fetch_range(
    session: requests.Session,
    start: dt.datetime,
    end: dt.datetime,
    *,
    delay: float,
) -> List[Dict]:
    """
    Fetch CVE entries published within [start, end).

    Returns a list of vulnerability dictionaries as provided by the NVD API.
    """

    results: List[Dict] = []
    start_iso = start.strftime("%Y-%m-%dT%H:%M:%S.000")
    end_iso = (end - dt.timedelta(milliseconds=1)).strftime("%Y-%m-%dT%H:%M:%S.000")
    start_index = 0

    while True:
        params = {
            "pubStartDate": start_iso,
            "pubEndDate": end_iso,
            "startIndex": start_index,
            "resultsPerPage": MAX_RESULTS_PER_REQUEST,
        }
        response = session.get(API_URL, params=params, timeout=60)
        if response.status_code != 200:
            raise RuntimeError(f"NVD API error {response.status_code}: {response.text}")
        payload = response.json()
        batch = payload.get("vulnerabilities", [])
        if not batch:
            break
        results.extend(batch)
        total = payload.get("totalResults", len(batch))
        start_index += len(batch)
        if start_index >= total:
            break
        time.sleep(delay)
    return results


def extract_record(entry: Dict) -> Optional[Dict[str, str]]:
    cve = entry.get("cve", {})
    cve_id = cve.get("id")
    published = cve.get("published") or entry.get("published")
    descriptions = cve.get("descriptions", [])
    description = ""
    for desc in descriptions:
        if desc.get("lang", "").lower() == "en" and desc.get("value"):
            description = desc["value"].strip()
            break
    if not cve_id or not description or description.lower() in {"reserved", "rejected"}:
        return None
    metrics = cve.get("metrics", {})
    cvss_v31 = metrics.get("cvssMetricV31") or []
    severity = None
    for metric in cvss_v31:
        data = metric.get("cvssData", {})
        base_severity = data.get("baseSeverity")
        if base_severity in ALLOWED_SEVERITIES:
            severity = base_severity
            break
    if severity is None:
        return None
    return {
        "cve_id": cve_id,
        "published": (published or "").split("T")[0],
        "description": description.replace("\n", " "),
        "severity": severity,
    }


def write_csv(records: Iterable[Dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ("cve_id", "published", "description", "severity")
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def fetch_cves_to_csv(
    start_year: int,
    end_year: int,
    output_path: Path,
    *,
    api_key: Optional[str] = None,
    delay: float = DEFAULT_DELAY,
    window_days: int = 30,
    sample_size: Optional[int] = None,
    seed: int = 42,
) -> int:
    """Programmatically fetch CVEs and persist them as CSV.

    Returns the number of written rows.
    """

    if end_year < start_year:
        raise ValueError("end_year must be >= start_year")

    session = build_session(api_key)
    collected: Dict[str, Dict[str, str]] = {}
    for year in range(start_year, end_year + 1):
        year_start = dt.datetime(year, 1, 1)
        year_end = dt.datetime(year + 1, 1, 1)
        LOGGER.info("Fetching CVEs for %s", year)
        for window_start, window_end in daterange(year_start, year_end, window_days):
            try:
                batch = fetch_range(session, window_start, window_end, delay=delay)
            except Exception as exc:  # pragma: no cover - network failure
                LOGGER.error("Failed to fetch %s – %s", window_start.date(), exc)
                time.sleep(delay * 2)
                continue
            for entry in batch:
                record = extract_record(entry)
                if record is None:
                    continue
                collected[record["cve_id"]] = record
            time.sleep(delay)

    if not collected:
        raise RuntimeError("No CVEs collected – check parameters or API availability.")

    records = list(collected.values())
    if sample_size is not None and sample_size < len(records):
        random.seed(seed)
        records = random.sample(records, sample_size)
        LOGGER.info("Sampling %s CVEs out of %s collected (seed=%s)", sample_size, len(collected), seed)
    LOGGER.info("Collected %s unique CVEs. Writing %s", len(records), output_path)
    write_csv(records, output_path)
    return len(records)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        total = fetch_cves_to_csv(
            start_year=args.start_year,
            end_year=args.end_year,
            output_path=args.output,
            api_key=args.api_key,
            delay=args.delay,
            window_days=args.window_days,
            sample_size=args.sample_size,
            seed=args.seed,
        )
    except Exception as exc:  # pragma: no cover - CLI surface
        raise SystemExit(str(exc))

    LOGGER.info("Done. Wrote %s rows to %s", total, args.output)


if __name__ == "__main__":
    main()
