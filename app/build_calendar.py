#!/usr/bin/env python3
"""Build an iCal feed for EKAC lap lanes, with optional debug artifacts.

- Discovers the latest Pool Schedule PDF (Aquatics page; falls back to fixed ID).
- Parses by weekday columns, clustering words into visual rows (robust to token splits).
- Emits iCal events in UTC and (if DEBUG=1) writes debug.html / debug.json.

Env:
  DEBUG=1       -> write debug artifacts to public/
  ROW_TOL=3.5   -> row-clustering tolerance in pixels (float)
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, List, Tuple

import pdfplumber
import requests
from bs4 import BeautifulSoup
from dateutil import parser as dtparse
from icalendar import Calendar, Event

AQUATICS_URL = "https://www.brooklinerec.com/150/Aquatics-Center"
FALLBACK_PDF = "https://www.brooklinerec.com/DocumentCenter/View/4404/Pool-Schedule?bidId="

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

DEBUG = os.environ.get("DEBUG", "0") == "1"
try:
    ROW_TOL = float(os.environ.get("ROW_TOL", "3.5"))
except ValueError:
    ROW_TOL = 3.5


@dataclass(frozen=True)
class Block:
    start: datetime  # UTC
    end: datetime    # UTC
    lanes_label: str
    source_url: str
    page: int
    day: str
    row_text: str


def discover_latest_pdf() -> str:
    """Find the latest Pool Schedule PDF URL with a robust fallback."""
    try:
        resp = requests.get(AQUATICS_URL, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            text = (a.get_text() or "").lower()
            if ("/DocumentCenter/View/" in href) and ("pool" in text and "schedule" in text):
                return requests.compat.urljoin(AQUATICS_URL, href)
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "/DocumentCenter/View/" in href and "Pool" in href and "Schedule" in href:
                return requests.compat.urljoin(AQUATICS_URL, href)
    except Exception:
        pass
    return FALLBACK_PDF


def month_span_to_dates(header: str, now_utc: datetime) -> Tuple[datetime, datetime] | None:
    """Parse 'Pool Schedule for August 4-10' -> (start_date, end_date) at midnight UTC."""
    m = re.search(r"Pool Schedule\s*(?:for)?\s+([A-Za-z]+)\s+(\d{1,2})\s*[-–]\s*(\d{1,2})", header)
    if not m:
        return None
    month, d1, d2 = m.group(1), int(m.group(2)), int(m.group(3))
    candidates = []
    for year in (now_utc.year - 1, now_utc.year, now_utc.year + 1):
        s = dtparse.parse(f"{month} {d1}, {year}").date()
        e = dtparse.parse(f"{month} {d2}, {year}").date()
        candidates.append((s, e, abs((now_utc.date() - s).days)))
    s, e, _ = min(candidates, key=lambda t: t[2])
    return (
        datetime(s.year, s.month, s.day, tzinfo=timezone.utc),
        datetime(e.year, e.month, e.day, tzinfo=timezone.utc),
    )


def to_utc_local_eastern(local_date: datetime, hh: int, mm: int) -> datetime:
    """Convert America/New_York local time to UTC for ICS."""
    from zoneinfo import ZoneInfo

    eastern = ZoneInfo("America/New_York")
    local_dt = datetime(local_date.year, local_date.month, local_date.day, hh, mm, tzinfo=eastern)
    return local_dt.astimezone(timezone.utc)


def time_token_to_24h(tstr: str) -> tuple[int, int]:
    """Accept '7a', '7 am', '7:15a', '7:15 pm', etc."""
    s = tstr.strip().lower().replace(" ", "")
    m = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?(a|p|am|pm)", s)
    if not m:
        raise ValueError(f"Bad time token: {tstr}")
    hh = int(m.group(1))
    mm = int(m.group(2) or 0)
    ap = m.group(3)
    if ap.startswith("a"):
        hh = 0 if hh == 12 else hh
    else:
        hh = 12 if hh == 12 else hh + 12
    return hh, mm


def _rows_from_words(day_words: list[dict], tol: float = ROW_TOL) -> list[list[dict]]:
    """Cluster tokens into visual rows by y ('top') with tolerance."""
    if not day_words:
        return []
    ws = sorted(day_words, key=lambda w: (w["top"], w["x0"]))
    rows: list[list[dict]] = []
    cur: list[dict] = [ws[0]]
    cur_y = ws[0]["top"]
    for w in ws[1:]:
        if abs(w["top"] - cur_y) <= tol:
            cur.append(w)
        else:
            rows.append(cur)
            cur = [w]
            cur_y = w["top"]
    rows.append(cur)
    for r in rows:
        r.sort(key=lambda w: w["x0"])
    return rows


def group_by_day_columns(words: list[dict]) -> tuple[dict[str, list[dict]], dict]:
    """Assign tokens to weekday columns. Fall back to 7 equal bands if headers missing."""
    name_map = {
        "monday": "Monday", "mon": "Monday",
        "tuesday": "Tuesday", "tue": "Tuesday", "tues": "Tuesday",
        "wednesday": "Wednesday", "wed": "Wednesday",
        "thursday": "Thursday", "thu": "Thursday", "thur": "Thursday", "thurs": "Thursday",
        "friday": "Friday", "fri": "Friday",
        "saturday": "Saturday", "sat": "Saturday",
        "sunday": "Sunday", "sun": "Sunday",
    }
    headers: list[tuple[str, float]] = []
    for w in words:
        key = name_map.get(w["text"].strip().lower())
        if key:
            headers.append((key, (w["x0"] + w["x1"]) / 2.0))

    debug = {"headers_raw": headers}
    best: dict[str, float] = {}
    for name, xmid in sorted(headers, key=lambda t: t[1]):
        if name not in best:
            best[name] = xmid

    bounds: list[tuple[str, float, float]] = []
    if len(best) >= 7:
        ordered = [
            ("Monday", best["Monday"]), ("Tuesday", best["Tuesday"]), ("Wednesday", best["Wednesday"]),
            ("Thursday", best["Thursday"]), ("Friday", best["Friday"]), ("Saturday", best["Saturday"]),
            ("Sunday", best["Sunday"]),
        ]
        for i, (day, xmid) in enumerate(ordered):
            left = -1e9 if i == 0 else (ordered[i - 1][1] + xmid) / 2.0
            right = 1e9 if i == 6 else (xmid + ordered[i + 1][1]) / 2.0
            bounds.append((day, left, right))
        debug["mode"] = "headers"
    else:
        # Fallback: split the page width into 7 equal vertical bands.
        if not words:
            minx, maxx = 0.0, 1000.0
        else:
            minx = min(w["x0"] for w in words)
            maxx = max(w["x1"] for w in words)
        step = (maxx - minx) / 7.0 if maxx > minx else 100.0
        x = minx
        for i, day in enumerate(DAYS):
            left = x
            right = maxx if i == 6 else x + step
            bounds.append((day, left, right))
            x += step
        debug["mode"] = "equal-bands"

    debug["bounds"] = bounds

    columns: dict[str, list[dict]] = {d: [] for d in DAYS}
    for w in words:
        xmid = (w["x0"] + w["x1"]) / 2.0
        for day, left, right in bounds:
            if left <= xmid <= right:
                columns[day].append(w)
                break
    for d in columns:
        columns[d].sort(key=lambda w: (w["top"], w["x0"]))
    return columns, debug


def extract_blocks_for_day(
    day_words: list[dict],
    local_date: datetime,
) -> Iterable[tuple[datetime, datetime, str, str]]:
    """Yield (start_utc, end_utc, lanes_label, row_text) using row-based scanning."""
    time_re = re.compile(
        r"(?P<s>\d{1,2}(?::\d{2})?\s*(?:a|p|am|pm))\s*[-–]\s*(?P<e>\d{1,2}(?::\d{2})?\s*(?:a|p|am|pm))",
        re.IGNORECASE,
    )
    lanes_re = re.compile(r"(?:^|\D)(?P<n1>\d{1,2})(?:\s*[–-]\s*(?P<n2>\d{1,2}))?(?:\D|$)")

    for row in _rows_from_words(day_words, tol=ROW_TOL):
        row_text = " ".join(w["text"] for w in row)
        m = time_re.search(row_text)
        if not m:
            continue
        # Prefer a lanes number that appears after the time range
        lanes = None
        post = row_text[m.end():]
        for m2 in lanes_re.finditer(post):
            n1 = int(m2.group("n1"))
            n2 = m2.group("n2")
            lanes = f"{n1}–{int(n2)}" if n2 else f"{n1}"
            break
        if not lanes:
            continue

        sh, sm = time_token_to_24h(m.group("s"))
        eh, em = time_token_to_24h(m.group("e"))
        st_utc = to_utc_local_eastern(local_date, sh, sm)
        en_utc = to_utc_local_eastern(local_date, eh, em)
        if en_utc <= st_utc:
            en_utc += timedelta(days=1)
        yield st_utc, en_utc, lanes, row_text


def parse_pdf(url: str) -> tuple[list[Block], dict]:
    """Download and parse the PDF into lap-lane blocks; collect debug info."""
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    now_utc = datetime.now(timezone.utc)
    blocks: list[Block] = []
    debug: dict = {"pdf": url, "pages": []}

    with pdfplumber.open(io.BytesIO(r.content)) as pdf:
        for page_index, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            header = next((ln for ln in text.splitlines() if "Pool Schedule" in ln), "")
            span = month_span_to_dates(header, now_utc)
            week_start = span[0] if span else now_utc
            while week_start.weekday() != 0:
                week_start -= timedelta(days=1)

            words = page.extract_words(keep_blank_chars=False, use_text_flow=True)
            columns, dbg_cols = group_by_day_columns(words)

            page_dbg = {
                "page": page_index,
                "header": header,
                "week_start_utc": week_start.isoformat(),
                "column_mode": dbg_cols.get("mode"),
                "bounds": dbg_cols.get("bounds"),
                "headers_raw": dbg_cols.get("headers_raw"),
                "days": {},
            }

            for offset, day in enumerate(DAYS):
                local_date = week_start + timedelta(days=offset)
                rows = [" ".join(w["text"] for w in row) for row in _rows_from_words(columns.get(day, []))]
                page_dbg["days"][day] = {"row_count": len(rows), "rows": rows, "matches": []}

                for st_utc, en_utc, label, row_text in extract_blocks_for_day(columns.get(day, []), local_date):
                    blocks.append(
                        Block(
                            start=st_utc,
                            end=en_utc,
                            lanes_label=label,
                            source_url=url,
                            page=page_index,
                            day=day,
                            row_text=row_text,
                        )
                    )
                    page_dbg["days"][day]["matches"].append(
                        {
                            "start": st_utc.isoformat(),
                            "end": en_utc.isoformat(),
                            "lanes": label,
                            "row_text": row_text,
                        }
                    )

            debug["pages"].append(page_dbg)

    return blocks, debug


def make_calendar(blocks: Iterable[Block]) -> Calendar:
    """Create the iCal, one VEVENT per block."""
    cal = Calendar()
    cal.add("prodid", "-//Brookline EKAC Lanes//bhs-pool-calendar//EN")
    cal.add("version", "2.0")
    cal.add("x-wr-calname", "Brookline EKAC Lap Lanes")
    cal.add("x-wr-timezone", "America/New_York")
    for b in blocks:
        ev = Event()
        ev.add("dtstart", b.start)
        ev.add("dtend", b.end)
        ev.add("summary", f"Lap lanes: {b.lanes_label.replace('-', '–')} open")
        ev.add("location", "Evelyn Kirrane Aquatics Center, 60 Tappan St, Brookline, MA 02446")
        ev.add("description", f"Page {b.page}, {b.day}\nRow: {b.row_text}\nSource: {b.source_url}")
        uid_raw = f"{b.start.isoformat()}|{b.end.isoformat()}|{b.lanes_label}"
        ev.add("uid", f"ekac-{hashlib.sha1(uid_raw.encode('utf-8')).hexdigest()}@bhs-pool-calendar")
        cal.add_component(ev)
    return cal


def _write_debug(debug: dict, blocks: list[Block]) -> None:
    """Write debug.json + debug.html to public/."""
    os.makedirs("public", exist_ok=True)
    with open("public/debug.json", "w", encoding="utf-8") as f:
        json.dump({"debug": debug, "event_count": len(blocks)}, f, indent=2)

    lines = []
    lines.append("<html><head><meta charset='utf-8'><title>EKAC Debug</title>")
    lines.append(
        "<style>body{font:14px/1.4 -apple-system,Segoe UI,Roboto,Helvetica,Arial}"
        "pre{white-space:pre-wrap;word-break:break-word}"
        "details{margin:8px 0}summary{font-weight:600}"
        "code{background:#f2f2f2;padding:1px 3px;border-radius:3px}"
        ".ok{color:#0a0}.warn{color:#b80}.bad{color:#b00}</style></head><body>"
    )
    lines.append(f"<h2>PDF: <code>{debug.get('pdf')}</code></h2>")
    lines.append(f"<p>Total parsed events: <b>{len(blocks)}</b></p>")
    for page in debug.get("pages", []):
        lines.append(f"<h3>Page {page['page']}</h3>")
        lines.append(f"<p>Header: <code>{page['header']}</code></p>")
        lines.append(
            f"<p>Week start (UTC): <code>{page['week_start_utc']}</code> | "
            f"Columns mode: <code>{page['column_mode']}</code></p>"
        )
        lines.append("<details><summary>Column bounds & headers</summary><pre>")
        lines.append(json.dumps({"bounds": page.get("bounds"), "headers_raw": page.get("headers_raw")}, indent=2))
        lines.append("</pre></details>")
        for day in DAYS:
            d = page["days"].get(day, {})
            cls = "ok" if d.get("matches") else "bad"
            lines.append(
                f"<details open><summary class='{cls}'>{day}: "
                f"{len(d.get('matches', []))} matches | {d.get('row_count', 0)} rows</summary>"
            )
            lines.append("<pre>Rows:\n" + "\n".join(d.get("rows", [])) + "</pre>")
            if d.get("matches"):
                lines.append("<pre>Matches:\n" + json.dumps(d["matches"], indent=2) + "</pre>")
            lines.append("</details>")
    lines.append("</body></html>")
    with open("public/debug.html", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> int:
    pdf_url = discover_latest_pdf()
    blocks, debug = parse_pdf(pdf_url)

    os.makedirs("public", exist_ok=True)
    cal = make_calendar(blocks)
    with open("public/ekac.ics", "wb") as f:
        f.write(cal.to_ical())

    if DEBUG:
        _write_debug(debug, blocks)

    print(f"Wrote public/ekac.ics with {len(blocks)} events from {pdf_url}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
