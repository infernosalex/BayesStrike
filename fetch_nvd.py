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


def main() -> None:
    args = parse_args()
    if args.end_year < args.start_year:
        raise SystemExit("end-year must be >= start-year")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    session = build_session(args.api_key)

    collected: Dict[str, Dict[str, str]] = {}
    for year in range(args.start_year, args.end_year + 1):
        year_start = dt.datetime(year, 1, 1)
        year_end = dt.datetime(year + 1, 1, 1)
        LOGGER.info("Fetching CVEs for %s", year)
        for window_start, window_end in daterange(year_start, year_end, args.window_days):
            try:
                batch = fetch_range(session, window_start, window_end, delay=args.delay)
            except Exception as exc:  # pragma: no cover - network failure
                LOGGER.error("Failed to fetch %s – %s", window_start.date(), exc)
                time.sleep(args.delay * 2)
                continue
            for entry in batch:
                record = extract_record(entry)
                if record is None:
                    continue
                collected[record["cve_id"]] = record
            time.sleep(args.delay)

    if not collected:
        raise SystemExit("No CVEs collected – check parameters or API availability.")

    LOGGER.info("Collected %s unique CVEs. Writing %s", len(collected), args.output)
    write_csv(collected.values(), args.output)
    LOGGER.info("Done.")


if __name__ == "__main__":
    main()

