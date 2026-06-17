#!/usr/bin/env python3
"""Scrape CAIE 9709 Mathematics papers from PastPapers.co and emit exam-bank input records.

This is intentionally conservative:
- It keeps only CAIE Mathematics 9709 PDFs.
- It keeps only question papers and mark schemes.
- It keeps only PM1, PM3, S1, and M1.
- It ignores years before 2008.
- It handles the old Statistics 1 component change: pre-2020 S1 is Paper 6; 2020+
  S1 is Paper 5.

Typical usage:

    python -m exam_bank.cli ingress pastpapers-co \
      --input data/exam_bank_input.jsonl \
      --output data/exam_bank_input.expanded.jsonl \
      --min-year 2008 \
      --max-year 2025

If --input is omitted, the script writes only the scraped records.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urldefrag, urljoin, urlparse
from urllib.request import Request, urlopen

BASE_URL = "https://pastpapers.co/caie/a-level/mathematics-9709"
SOURCE_NAME = "pastpapers.co"
SYLLABUS = "9709"

# Cambridge file names look like:
#   9709_w08_qp_1.pdf
#   9709_s25_ms_52.pdf
PDF_RE = re.compile(
    r"(?P<syllabus>9709)_(?P<series_code>[msw])(?P<yy>\d{2})_(?P<doc_type>qp|ms)_(?P<component>\d{1,2})\.pdf$",
    re.IGNORECASE,
)

# Secondary extraction for links serialized inside JS/JSON instead of normal <a href> tags.
PATH_LINK_RE = re.compile(
    r"(?P<href>(?:https?://pastpapers\.co)?/caie/a-level/mathematics-9709/[^\"'<>\\\s]+)",
    re.IGNORECASE,
)
RELATIVE_PDF_RE = re.compile(
    r"(?<![A-Za-z0-9_/.-])(?P<href>9709_[msw]\d{2}_(?:qp|ms)_\d{1,2}\.pdf)",
    re.IGNORECASE,
)

SERIES_NAMES = {
    "m": "March",
    "s": "May/June",
    "w": "Oct/Nov",
}

PAPER_LABELS = {
    "pure_math_1": "Pure Mathematics 1",
    "pure_math_3": "Pure Mathematics 3",
    "statistics_1": "Probability & Statistics 1",
    "mechanics_1": "Mechanics 1",
}

PAPER_ORDER = {
    "pure_math_1": 0,
    "pure_math_3": 1,
    "statistics_1": 2,
    "mechanics_1": 3,
}

SESSION_ORDER = {"m": 0, "s": 1, "w": 2}


@dataclass(frozen=True)
class PaperResource:
    """One discovered PDF resource before QP/MS pairing."""

    syllabus: str
    year: int
    series_code: str
    session: str
    doc_type: str  # qp or ms
    component: str
    paper: str
    paper_name: str
    url: str
    source_page: str | None = None


class LinkExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() not in {"a", "link", "script"}:
            return
        for key, value in attrs:
            if key.lower() in {"href", "src"} and value:
                self.links.append(value)


def fetch_text(url: str, *, timeout: float = 20.0, retries: int = 2) -> str | None:
    """Fetch a page as text. Returns None for 404/connection failures."""

    headers = {
        "User-Agent": (
            "Mozilla/5.0 compatible; exam-bank-ingress/1.0; "
            "+https://pastpapers.co/caie/a-level/mathematics-9709"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            req = Request(url, headers=headers)
            with urlopen(req, timeout=timeout) as response:  # nosec: target URL is fixed by caller filters
                content_type = response.headers.get("content-type", "")
                raw = response.read()
                if "pdf" in content_type.lower() or urlparse(url).path.lower().endswith(".pdf"):
                    # We only need PDF URLs, not binary content.
                    return ""
                charset_match = re.search(r"charset=([^;]+)", content_type, re.IGNORECASE)
                charset = charset_match.group(1) if charset_match else "utf-8"
                return raw.decode(charset, errors="replace")
        except HTTPError as exc:
            if exc.code in {404, 410}:
                return None
            last_error = exc
        except (URLError, TimeoutError, OSError) as exc:
            last_error = exc
        if attempt < retries:
            time.sleep(0.5 * (attempt + 1))
    print(f"warning: failed to fetch {url}: {last_error}", file=sys.stderr)
    return None


def normalize_link(current_url: str, href: str) -> str | None:
    """Normalize and constrain links to the CAIE 9709 PastPapers.co subtree."""

    href = unescape(href).replace("\\/", "/").strip()
    if not href or href.startswith(("#", "mailto:", "javascript:")):
        return None
    join_base = current_url
    if not join_base.endswith("/") and not urlparse(join_base).path.lower().endswith(".pdf"):
        # PastPapers.co session pages are directory-like routes that are often served
        # without a trailing slash. Treat relative PDFs as children of the session route.
        join_base += "/"
    absolute, _fragment = urldefrag(urljoin(join_base, href))
    parsed = urlparse(absolute)
    if parsed.scheme not in {"http", "https"}:
        return None
    if parsed.netloc.lower() != "pastpapers.co":
        return None
    path = parsed.path.rstrip("/")
    base_path = urlparse(BASE_URL).path.rstrip("/")
    if not path.startswith(base_path):
        return None
    return absolute.rstrip("/")


def extract_links(current_url: str, html: str) -> set[str]:
    """Extract normal href links plus links embedded in serialized JS/JSON."""

    html = html.replace("\\/", "/")
    extractor = LinkExtractor()
    try:
        extractor.feed(html)
    except Exception:
        # HTMLParser is forgiving, but malformed script blobs should not kill discovery.
        pass

    raw_links = set(extractor.links)
    raw_links.update(match.group("href") for match in PATH_LINK_RE.finditer(html))
    raw_links.update(match.group("href") for match in RELATIVE_PDF_RE.finditer(html))

    normalized: set[str] = set()
    for href in raw_links:
        link = normalize_link(current_url, href)
        if link:
            normalized.add(link)
    return normalized


def parse_two_digit_year(yy: str) -> int:
    # All target years are 2008+. Treat 00-79 as 2000s to avoid surprises when the
    # site eventually adds future sessions; 80-99 would be legacy 1900s and filtered.
    value = int(yy)
    return 2000 + value if value <= 79 else 1900 + value


def classify_component(component: str, year: int) -> str | None:
    """Map a CAIE 9709 component number to the requested paper bucket."""

    component = component.lstrip("0") or "0"

    if component in {"1", "11", "12", "13"}:
        return "pure_math_1"
    if component in {"3", "31", "32", "33"}:
        return "pure_math_3"
    if component in {"4", "41", "42", "43"}:
        return "mechanics_1"

    # 9709 changed the Statistics component numbering in 2020.
    # Before 2020, Probability & Statistics 1 was Paper 6; from 2020 onward it is Paper 5.
    if year >= 2020 and component in {"5", "51", "52", "53"}:
        return "statistics_1"
    if year < 2020 and component in {"6", "61", "62", "63"}:
        return "statistics_1"

    return None


def parse_pdf_resource(url: str, *, source_page: str | None = None) -> PaperResource | None:
    """Parse and filter a PastPapers.co PDF URL."""

    filename = unquote(urlparse(url).path.split("/")[-1]).lower()
    match = PDF_RE.search(filename)
    if not match:
        return None

    year = parse_two_digit_year(match.group("yy"))
    component = match.group("component").lstrip("0") or "0"
    paper = classify_component(component, year)
    if paper is None:
        return None

    series_code = match.group("series_code").lower()
    return PaperResource(
        syllabus=match.group("syllabus"),
        year=year,
        series_code=series_code,
        session=SERIES_NAMES[series_code],
        doc_type=match.group("doc_type").lower(),
        component=component,
        paper=paper,
        paper_name=PAPER_LABELS[paper],
        url=url,
        source_page=source_page,
    )


def session_page_candidates(min_year: int, max_year: int) -> Iterator[str]:
    """Known PastPapers.co session-route shapes.

    The site currently exposes newer sessions at e.g. /2025-may-june and older
    sessions at e.g. /2008/2008-jun. We seed both route families and let HTTP 404
    filtering remove non-existent pages.
    """

    modern_slugs = ["{year}-march", "{year}-may-june", "{year}-oct-nov"]
    legacy_slugs = ["{year}-mar", "{year}-jun", "{year}-nov"]
    extra_legacy_slugs = ["{year}-march", "{year}-may-june", "{year}-oct-nov"]

    for year in range(min_year, max_year + 1):
        for pattern in modern_slugs:
            yield f"{BASE_URL}/{pattern.format(year=year)}"
        for pattern in legacy_slugs + extra_legacy_slugs:
            slug = pattern.format(year=year)
            yield f"{BASE_URL}/{year}/{slug}"


def discover_resources(
    *,
    base_url: str = BASE_URL,
    min_year: int = 2008,
    max_year: int = 2025,
    max_depth: int = 2,
    delay_seconds: float = 0.2,
    fetcher: Callable[[str], str | None] | None = None,
    include_session_candidates: bool = True,
) -> list[PaperResource]:
    """Crawl the source page and return filtered paper resources."""

    fetcher = fetcher or fetch_text
    seeds = [base_url]
    if include_session_candidates:
        seeds.extend(session_page_candidates(min_year, max_year))

    queue: deque[tuple[str, int]] = deque((url, 0) for url in dict.fromkeys(seeds))
    visited: set[str] = set()
    resources: dict[tuple[str, str], PaperResource] = {}

    while queue:
        url, depth = queue.popleft()
        if url in visited:
            continue
        visited.add(url)

        parsed = parse_pdf_resource(url, source_page=None)
        if parsed and min_year <= parsed.year <= max_year:
            resources[(parsed.doc_type, parsed.url)] = parsed
            continue

        if urlparse(url).path.lower().endswith(".pdf") or depth > max_depth:
            continue

        html = fetcher(url)
        if html is None:
            continue
        if delay_seconds:
            time.sleep(delay_seconds)

        for link in extract_links(url, html):
            pdf_resource = parse_pdf_resource(link, source_page=url)
            if pdf_resource:
                if min_year <= pdf_resource.year <= max_year:
                    resources[(pdf_resource.doc_type, pdf_resource.url)] = pdf_resource
                continue
            if depth < max_depth and link not in visited:
                queue.append((link, depth + 1))

    return sorted(resources.values(), key=resource_sort_key)


def resource_sort_key(resource: PaperResource) -> tuple[int, int, int, int, int, str]:
    return (
        resource.year,
        SESSION_ORDER.get(resource.series_code, 99),
        PAPER_ORDER.get(resource.paper, 99),
        int(resource.component),
        0 if resource.doc_type == "qp" else 1,
        resource.url,
    )


def build_exam_records(resources: Iterable[PaperResource]) -> list[dict[str, Any]]:
    """Pair question papers and mark schemes into exam-bank input records."""

    grouped: dict[tuple[int, str, str, str], dict[str, Any]] = {}

    for resource in resources:
        key = (resource.year, resource.series_code, resource.paper, resource.component)
        record = grouped.setdefault(
            key,
            {
                "id": make_record_id(resource),
                "board": "CAIE",
                "qualification": "A Level",
                "subject": "Mathematics",
                "syllabus": resource.syllabus,
                "paper": resource.paper,
                "paper_name": resource.paper_name,
                "component": resource.component,
                "variant": variant_from_component(resource.component),
                "year": resource.year,
                "session": resource.session,
                "session_code": resource.series_code,
                "source": SOURCE_NAME,
                "source_page": resource.source_page,
                "question_paper_url": None,
                "mark_scheme_url": None,
            },
        )
        if resource.source_page and not record.get("source_page"):
            record["source_page"] = resource.source_page
        if resource.doc_type == "qp":
            record["question_paper_url"] = resource.url
        elif resource.doc_type == "ms":
            record["mark_scheme_url"] = resource.url

    records = list(grouped.values())
    records.sort(key=record_sort_key)
    return records


def make_record_id(resource: PaperResource) -> str:
    return f"caie-9709-{resource.paper}-{resource.series_code}{str(resource.year)[-2:]}-{resource.component}"


def variant_from_component(component: str) -> str | None:
    # Component 12 means paper 1, variant 2. Legacy component 1 has no variant.
    return component[-1] if len(component) == 2 else None


def record_sort_key(record: dict[str, Any]) -> tuple[int, int, int, int, str]:
    return (
        int(record.get("year", 0)),
        SESSION_ORDER.get(str(record.get("session_code", "")), 99),
        PAPER_ORDER.get(str(record.get("paper", "")), 99),
        int(record.get("component") or 0),
        str(record.get("id", "")),
    )


def load_input(path: Path) -> tuple[list[dict[str, Any]], str, dict[str, Any] | None, str | None]:
    """Load JSONL, JSON array, or a JSON object containing a record-list key.

    Returns: records, format_name, original_object, list_key.
    """

    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if not stripped:
        return [], "jsonl", None, None

    if path.suffix.lower() == ".jsonl" or not stripped.startswith(("[", "{")):
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
        return records, "jsonl", None, None

    data = json.loads(text)
    if isinstance(data, list):
        return data, "json-list", None, None

    if isinstance(data, dict):
        for key in ("records", "papers", "exams", "input", "items"):
            if isinstance(data.get(key), list):
                return data[key], "json-object", data, key
        # Fall back to the first list of objects.
        for key, value in data.items():
            if isinstance(value, list) and all(isinstance(item, dict) for item in value):
                return value, "json-object", data, key

    raise ValueError(f"Unsupported input shape in {path}. Expected JSONL, JSON array, or JSON object with a list field.")


def record_identity(record: dict[str, Any]) -> tuple[Any, ...]:
    if record.get("id"):
        return ("id", record["id"])
    return (
        "paper",
        record.get("board"),
        record.get("syllabus"),
        record.get("paper"),
        record.get("year"),
        record.get("session_code"),
        record.get("component"),
    )


def merge_records(existing: Iterable[dict[str, Any]], scraped: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Preserve existing records and append/update scraped records by stable identity."""

    merged: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in existing:
        merged[record_identity(record)] = dict(record)

    for record in scraped:
        key = record_identity(record)
        if key not in merged:
            merged[key] = dict(record)
            continue
        # Do not clobber existing values; fill missing URL/metadata fields only.
        target = merged[key]
        for field, value in record.items():
            if target.get(field) in (None, "", []):
                target[field] = value

    result = list(merged.values())
    result.sort(key=lambda record: (int(record.get("year", 9999) or 9999), str(record.get("id", ""))))
    return result


def write_output(
    path: Path,
    records: list[dict[str, Any]],
    *,
    format_name: str = "jsonl",
    original_object: dict[str, Any] | None = None,
    list_key: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if format_name == "jsonl":
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        return

    if format_name == "json-list":
        path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return

    if format_name == "json-object" and original_object is not None and list_key:
        original_object = dict(original_object)
        original_object[list_key] = records
        path.write_text(json.dumps(original_object, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return

    raise ValueError(f"Unsupported output format: {format_name}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=BASE_URL, help="PastPapers.co CAIE 9709 root page")
    parser.add_argument("--input", type=Path, help="Existing exam-bank input JSON/JSONL to merge into")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON/JSONL path")
    parser.add_argument("--min-year", type=int, default=2008)
    parser.add_argument("--max-year", type=int, default=2025)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--delay", type=float, default=0.2, help="Delay between HTTP page fetches")
    parser.add_argument(
        "--no-session-candidates",
        action="store_true",
        help="Only crawl links from --url; do not seed known year/session pages",
    )
    parser.add_argument(
        "--scraped-only",
        action="store_true",
        help="Ignore --input and write only scraped records",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.min_year < 2008:
        raise SystemExit("--min-year must be >= 2008 for this ingress")
    if args.max_year < args.min_year:
        raise SystemExit("--max-year must be >= --min-year")

    resources = discover_resources(
        base_url=args.url,
        min_year=args.min_year,
        max_year=args.max_year,
        max_depth=args.max_depth,
        delay_seconds=args.delay,
        include_session_candidates=not args.no_session_candidates,
    )
    scraped_records = build_exam_records(resources)

    if args.input and not args.scraped_only:
        existing_records, format_name, original_object, list_key = load_input(args.input)
        output_records = merge_records(existing_records, scraped_records)
    else:
        format_name, original_object, list_key = "jsonl", None, None
        output_records = scraped_records

    write_output(
        args.output,
        output_records,
        format_name=format_name,
        original_object=original_object,
        list_key=list_key,
    )

    print(
        f"discovered {len(resources)} resources; wrote {len(scraped_records)} scraped records "
        f"to {args.output}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
