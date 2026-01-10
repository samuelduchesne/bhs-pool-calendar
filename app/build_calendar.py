#!/usr/bin/env python3
"""Build iCal feeds for EKAC Lap, Shallow, and Dive from Google Sheets.

- Fetches the Pool Schedule from Google Sheets (as CSV).
- Parses the CSV to extract schedule information for each day of the week.
- Classifies each row into Lap / Shallow / Dive based on content.
- Extracts time ranges; for Lap rows it also extracts lane counts.
- Writes four feeds:
    public/ekac-lap.ics
    public/ekac-shallow.ics
    public/ekac-dive.ics
    public/ekac.ics   (combined)
- Writes public/index.html (landing page with "About/Disclaimer").
- If DEBUG=1, also writes public/debug.html and public/debug.json.

Env:
  DEBUG=1                 -> write debug artifacts
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, Literal

import requests
from dateutil import parser as dtparse
from icalendar import Calendar, Event

# Google Sheets URL - can be overridden with env var
SHEETS_ID = "1eYOnElx9lFYqS7XgjnhZrjDY5xWY_96RWftZ7TK1acU"
SHEETS_URL = f"https://docs.google.com/spreadsheets/d/{SHEETS_ID}/export?format=csv"
AQUATICS_URL = "https://www.brooklinerec.com/150/Aquatics-Center"

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

DEBUG = os.environ.get("DEBUG", "0") == "1"

Kind = Literal["lap", "shallow", "dive"]


@dataclass(frozen=True)
class Block:
    start: datetime  # UTC
    end: datetime  # UTC
    kind: Kind
    label: str  # "5", "4–5" for lap; "open" for shallow/dive
    source_url: str
    day: str
    row_text: str


SECTION_KEYS = {
    "lap": ["lap lane", "lap lanes", "lap swim", "lap"],
    "shallow": ["shallow pool", "shallow"],
    "dive": ["dive well", "diving well", "dive"],
}


def discover_sheet_gids() -> list[str]:
    """Discover all sheet GIDs in the Google Sheets document by fetching the HTML view."""
    html_url = f"https://docs.google.com/spreadsheets/d/{SHEETS_ID}/htmlview"
    try:
        resp = requests.get(html_url, timeout=30)
        resp.raise_for_status()
        # Extract all unique GIDs from the HTML
        gids = re.findall(r'gid=(\d+)', resp.text)
        # Return unique GIDs, preserving order
        seen = set()
        result = []
        for gid in gids:
            if gid not in seen:
                seen.add(gid)
                result.append(gid)
        return result
    except Exception as e:
        print(f"Warning: Could not discover sheet GIDs: {e}", file=sys.stderr)
        return []


def fetch_sheet_csv(gid: str | None = None) -> str:
    """Fetch a Google Sheet as CSV text.
    
    Args:
        gid: Optional sheet GID to fetch. If None, fetches the default sheet.
    """
    if gid:
        url = f"https://docs.google.com/spreadsheets/d/{SHEETS_ID}/export?format=csv&gid={gid}"
    else:
        url = os.environ.get("SHEETS_URL", SHEETS_URL)
    
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.text


def fetch_all_sheets() -> list[tuple[str, str]]:
    """Fetch all sheets from the Google Sheets document.
    
    Returns:
        List of (gid, csv_text) tuples, one for each sheet.
    """
    gids = discover_sheet_gids()
    
    if not gids:
        # Fallback to fetching the default sheet
        print("No GIDs found, falling back to default sheet", file=sys.stderr)
        return [("default", fetch_sheet_csv())]
    
    sheets = []
    for gid in gids:
        try:
            csv_text = fetch_sheet_csv(gid)
            sheets.append((gid, csv_text))
            print(f"Fetched sheet GID {gid}", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Could not fetch sheet GID {gid}: {e}", file=sys.stderr)
    
    return sheets


def parse_week_header(header_text: str, now_utc: datetime) -> tuple[datetime, datetime] | None:
    """Parse schedule header to determine the week range.
    
    Examples:
    - "Pool Schedule for August 4-10"
    - "Week of August 4-10, 2024"
    - "August 4-10"
    """
    # Pattern that captures: (month, day1, day2, year?)
    pattern = r"([A-Za-z]+)\s+(\d{1,2})\s*[-–]\s*(\d{1,2})(?:,?\s*(\d{4}))?"
    
    m = re.search(pattern, header_text, re.IGNORECASE)
    if not m:
        return None
    
    month_str = m.group(1)
    d1 = int(m.group(2))
    d2 = int(m.group(3))
    year_str = m.group(4)
    
    # If year not specified, find closest year
    if year_str is None:
        candidates = []
        for y in (now_utc.year - 1, now_utc.year, now_utc.year + 1):
            try:
                s = dtparse.parse(f"{month_str} {d1}, {y}").date()
                e = dtparse.parse(f"{month_str} {d2}, {y}").date()
                candidates.append((s, e, abs((now_utc.date() - s).days)))
            except (ValueError, dtparse.ParserError):
                continue
        if candidates:
            s, e, _ = min(candidates, key=lambda t: t[2])
            return (
                datetime(s.year, s.month, s.day, tzinfo=timezone.utc),
                datetime(e.year, e.month, e.day, tzinfo=timezone.utc),
            )
    else:
        try:
            year = int(year_str)
            s = dtparse.parse(f"{month_str} {d1}, {year}").date()
            e = dtparse.parse(f"{month_str} {d2}, {year}").date()
            return (
                datetime(s.year, s.month, s.day, tzinfo=timezone.utc),
                datetime(e.year, e.month, e.day, tzinfo=timezone.utc),
            )
        except (ValueError, dtparse.ParserError):
            pass
    
    return None


def to_utc_local_eastern(local_date: datetime, hh: int, mm: int) -> datetime:
    """Convert America/New_York local time to UTC (ICS stores UTC)."""
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


def classify_kind(text: str) -> Kind:
    """Classify a schedule entry as lap, shallow, or dive based on text content."""
    text_lower = text.lower()
    
    # Check for specific markers
    for kind_str, keys in SECTION_KEYS.items():
        for key in keys:
            if key in text_lower:
                # Cast to Kind type since we know kind_str is one of "lap", "shallow", or "dive"
                return kind_str  # type: ignore[return-value]
    
    # Default to lap if not specified
    return "lap"


def extract_lane_count(text: str) -> str:
    """Extract lane count from text like '4-5 lanes', '5 lanes', etc."""
    # Look for patterns like "4-5", "4–5", "5"
    m = re.search(r"(?<!\d)(\d{1,2})\s*[-–]\s*(\d{1,2})(?:\s*lanes?)?", text, re.IGNORECASE)
    if m:
        return f"{m.group(1)}–{m.group(2)}"
    
    m = re.search(r"(?<!\d)(\d{1,2})(?:\s*lanes?)(?!\d)", text, re.IGNORECASE)
    if m:
        return m.group(1)
    
    return "open"


def parse_time_range(text: str) -> tuple[int, int, int, int] | None:
    """Parse time range from text like '7am - 9am', '7:30am-9pm', etc.
    
    Returns (start_hour, start_min, end_hour, end_min) or None if not found.
    """
    time_re = re.compile(
        r"(\d{1,2}(?::\d{2})?\s*(?:a|p|am|pm))\s*[-–]\s*(\d{1,2}(?::\d{2})?\s*(?:a|p|am|pm))",
        re.IGNORECASE,
    )
    
    m = time_re.search(text)
    if not m:
        return None
    
    try:
        sh, sm = time_token_to_24h(m.group(1))
        eh, em = time_token_to_24h(m.group(2))
        return sh, sm, eh, em
    except ValueError:
        return None


def parse_csv_schedule(csv_text: str, now_utc: datetime) -> tuple[list[Block], dict]:
    """Parse the CSV schedule and extract blocks for each day."""
    blocks: list[Block] = []
    debug: dict = {"source": SHEETS_URL, "rows": []}
    
    lines = csv_text.strip().split('\n')
    reader = csv.reader(lines)
    rows = list(reader)
    
    if not rows:
        return blocks, debug
    
    # Try to find week range from first few rows
    week_start = now_utc
    for i in range(min(5, len(rows))):
        header_text = ' '.join(rows[i])
        span = parse_week_header(header_text, now_utc)
        if span:
            week_start = span[0]
            break
    
    # Ensure week_start is a Monday
    while week_start.weekday() != 0:
        week_start -= timedelta(days=1)
    
    debug["week_start_utc"] = week_start.isoformat()
    
    # Find header row with day names
    day_indices = {}
    header_row_idx = -1
    
    for i, row in enumerate(rows):
        for j, cell in enumerate(row):
            cell_lower = cell.strip().lower()
            # Map common day abbreviations to full names
            day_map = {
                "monday": "Monday", "mon": "Monday",
                "tuesday": "Tuesday", "tue": "Tuesday", "tues": "Tuesday",
                "wednesday": "Wednesday", "wed": "Wednesday",
                "thursday": "Thursday", "thu": "Thursday", "thur": "Thursday",
                "friday": "Friday", "fri": "Friday",
                "saturday": "Saturday", "sat": "Saturday",
                "sunday": "Sunday", "sun": "Sunday",
            }
            if cell_lower in day_map:
                day_name = day_map[cell_lower]
                if day_name not in day_indices:
                    day_indices[day_name] = j
                    header_row_idx = i
    
    debug["day_indices"] = day_indices
    debug["header_row_idx"] = header_row_idx
    
    if not day_indices or header_row_idx < 0:
        # No header row found, try to infer structure
        # Assume columns are: Time, Mon, Tue, Wed, Thu, Fri, Sat, Sun
        if len(rows) > 0 and len(rows[0]) >= 8:
            day_indices = {day: i+1 for i, day in enumerate(DAYS)}
            header_row_idx = 0
    
    # Parse data rows after header
    for i in range(header_row_idx + 1, len(rows)):
        row = rows[i]
        row_text = ' '.join(row)
        
        # Skip empty rows
        if not row_text.strip():
            continue
        
        debug["rows"].append({"row_num": i, "content": row, "row_text": row_text})
        
        # Extract time range from the row
        time_info = parse_time_range(row_text)
        if not time_info:
            continue
        
        sh, sm, eh, em = time_info
        
        # Process each day column
        for day_name, col_idx in day_indices.items():
            if col_idx >= len(row):
                continue
            
            cell_text = row[col_idx].strip()
            if not cell_text:
                continue
            
            # Calculate date for this day
            day_offset = DAYS.index(day_name)
            local_date = week_start + timedelta(days=day_offset)
            
            # Convert times to UTC
            st_utc = to_utc_local_eastern(local_date, sh, sm)
            en_utc = to_utc_local_eastern(local_date, eh, em)
            if en_utc <= st_utc:
                en_utc += timedelta(days=1)
            
            # Classify the kind of pool
            kind = classify_kind(cell_text)
            
            # Extract label (lane count for lap, "open" for others)
            if kind == "lap":
                label = extract_lane_count(cell_text)
                # Skip lap entries with no lane count
                if label == "open":
                    continue
            else:
                label = "open"
            
            block = Block(
                start=st_utc,
                end=en_utc,
                kind=kind,
                label=label,
                source_url=SHEETS_URL,
                day=day_name,
                row_text=row_text,
            )
            blocks.append(block)
    
    return blocks, debug


def make_calendar(blocks: list[Block], kind: Kind | None, calname: str) -> Calendar:
    """If kind is None, include all kinds."""
    cal = Calendar()
    cal.add("prodid", "-//Brookline EKAC Feeds//bhs-pool-calendar//EN")
    cal.add("version", "2.0")
    cal.add("x-wr-calname", calname)
    cal.add("x-wr-timezone", "America/New_York")

    for b in blocks:
        if kind and b.kind != kind:
            continue
        ev = Event()
        ev.add("dtstart", b.start)
        ev.add("dtend", b.end)
        if b.kind == "lap":
            ev.add("summary", f"Lap lanes: {b.label.replace('-', '–')} open")
        elif b.kind == "shallow":
            ev.add("summary", "Shallow pool: open")
        else:
            ev.add("summary", "Dive well: open")
        ev.add("location", "Evelyn Kirrane Aquatics Center, 60 Tappan St, Brookline, MA 02446")
        ev.add("description", f"{b.day} — {b.row_text}\nSource: {b.source_url}")
        uid_raw = f"{b.start.isoformat()}|{b.end.isoformat()}|{b.kind}|{b.label}"
        ev.add("uid", f"ekac-{hashlib.sha1(uid_raw.encode('utf-8')).hexdigest()}@bhs-pool-calendar")
        cal.add_component(ev)

    return cal


def _write_debug(debug: dict, blocks: list[Block]) -> None:
    """Write debug.json + debug.html to public/ for inspection."""
    os.makedirs("public", exist_ok=True)
    with open("public/debug.json", "w", encoding="utf-8") as f:
        json.dump({"debug": debug, "event_count": len(blocks)}, f, indent=2)

    lines = []
    lines.append(
        "<html><head><meta charset='utf-8'>"
        "<meta name='robots' content='noindex,nofollow'>"
        "<title>EKAC Debug</title>"
        "<style>body{font:14px/1.4 -apple-system,Segoe UI,Roboto,Helvetica,Arial}"
        "pre{white-space:pre-wrap;word-break:break-word}"
        "details{margin:8px 0}summary{font-weight:600}"
        "code{background:#f2f2f2;padding:1px 3px;border-radius:3px}"
        ".ok{color:#0a0}.warn{color:#b80}.bad{color:#b00}</style></head><body>"
    )
    lines.append("<h2>Multi-Sheet Debug</h2>")
    lines.append(f"<p>Total parsed events: <b>{len(blocks)}</b></p>")
    kind_counts = {"lap": 0, "shallow": 0, "dive": 0}
    for b in blocks:
        kind_counts[b.kind] += 1
    lines.append(f"<p>By kind: {kind_counts}</p>")
    
    # Display info for each sheet
    for i, sheet_info in enumerate(debug.get("sheets", [])):
        gid = sheet_info.get("gid", "unknown")
        blocks_count = sheet_info.get("blocks_count", 0)
        sheet_debug = sheet_info.get("debug", {})
        
        lines.append(f"<h3>Sheet {i+1}: GID {gid} ({blocks_count} events)</h3>")
        lines.append(f"<p>Source: <code>{sheet_debug.get('source', 'N/A')}</code></p>")
        lines.append(f"<p>Week start (UTC): <code>{sheet_debug.get('week_start_utc', 'N/A')}</code></p>")
        lines.append(f"<p>Day indices: <code>{sheet_debug.get('day_indices', {})}</code></p>")
        lines.append(f"<p>Header row: <code>{sheet_debug.get('header_row_idx', 'N/A')}</code></p>")
        
        lines.append(f"<h4>Parsed Rows (Sheet {i+1})</h4>")
        for row_info in sheet_debug.get("rows", []):
            lines.append(f"<details><summary>Row {row_info['row_num']}</summary>")
            lines.append(f"<pre>{row_info['row_text']}</pre>")
            lines.append("</details>")
    
    lines.append("<h3>All Generated Events</h3>")
    for b in blocks:
        lines.append(f"<p><b>{b.day}</b> - {b.kind}: {b.start.isoformat()} to {b.end.isoformat()} - {b.label}</p>")
    
    lines.append("</body></html>")
    with open("public/debug.html", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _write_index() -> None:
    """Landing page with About/Disclaimer and links to feeds."""
    os.makedirs("public", exist_ok=True)
    owner = os.getenv("GITHUB_REPOSITORY_OWNER", "")
    repo = os.getenv("GITHUB_REPOSITORY", "")
    html = f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>EKAC Pool Feeds</title>
<style>
 body{{font:16px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial;margin:24px;color:#222}}
 .card{{border:1px solid #e7e7e7;border-radius:8px;padding:16px;margin:12px 0}}
 .notice{{background:#fffbea;border-color:#ffe08a}}
 code{{background:#f5f5f5;padding:2px 4px;border-radius:4px}}
 a{{text-decoration:none}}
 h1{{margin:0 0 12px 0}} h2{{margin:12px 0}} p{{margin:8px 0}}
</style></head><body>

<div class="card notice">
  <h2>About this project</h2>
  <p>
    Unofficial, community-built calendars generated from the weekly Pool Schedule published by the
    Town of Brookline Recreation Department. For the official schedule, see
    <a href="{AQUATICS_URL}">the Aquatics Center page</a>.
  </p>
  <p>
    <b>Disclaimer:</b> No guarantees. Parsing can be wrong or out of date; always confirm with the facility.
    This is just a fun project using ChatGPT 5!
  </p>
  <p>Source code: <a href="https://github.com/{repo}">github.com/{repo}</a></p>
</div>

<h1>Evelyn Kirrane Aquatics Center — Pool Schedules</h1>
<p>Subscribe to one or more calendars:</p>

<div class="card">
  <h2>Lap Lanes</h2>
  <p>Shows lane availability (e.g., "Lap lanes: 4–5 open").</p>
  <p><a href="ekac-lap.ics"><b>HTTPS</b></a> &nbsp;|&nbsp;
     <a href="webcal://{owner}.github.io/bhs-pool-calendar/ekac-lap.ics"><b>webcal://</b></a></p>
</div>

<div class="card">
  <h2>Shallow Pool</h2>
  <p>Shallow pool open/closed times.</p>
  <p><a href="ekac-shallow.ics"><b>HTTPS</b></a> &nbsp;|&nbsp;
     <a href="webcal://{owner}.github.io/bhs-pool-calendar/ekac-shallow.ics"><b>webcal://</b></a></p>
</div>

<div class="card">
  <h2>Dive Well</h2>
  <p>Dive well open/closed times.</p>
  <p><a href="ekac-dive.ics"><b>HTTPS</b></a> &nbsp;|&nbsp;
     <a href="webcal://{owner}.github.io/bhs-pool-calendar/ekac-dive.ics"><b>webcal://</b></a></p>
</div>

<div class="card">
  <h2>All-in-one</h2>
  <p>Includes Lap, Shallow, and Dive in one feed.</p>
  <p><a href="ekac.ics"><b>HTTPS</b></a> &nbsp;|&nbsp;
     <a href="webcal://{owner}.github.io/bhs-pool-calendar/ekac.ics"><b>webcal://</b></a></p>
</div>

</body></html>"""
    with open("public/index.html", "w", encoding="utf-8") as f:
        f.write(html)


def main() -> int:
    now_utc = datetime.now(timezone.utc)
    
    # Fetch and parse all sheets from the Google Sheets document
    all_sheets = fetch_all_sheets()
    all_blocks = []
    all_debug = {"sheets": []}
    
    for gid, csv_text in all_sheets:
        blocks, debug = parse_csv_schedule(csv_text, now_utc)
        all_blocks.extend(blocks)
        all_debug["sheets"].append({
            "gid": gid,
            "blocks_count": len(blocks),
            "debug": debug
        })
        print(f"Parsed sheet GID {gid}: {len(blocks)} events", file=sys.stderr)
    
    # Remove duplicate blocks (same start, end, kind, label)
    # This can happen if there are overlapping schedules
    seen = set()
    unique_blocks = []
    for block in all_blocks:
        key = (block.start, block.end, block.kind, block.label)
        if key not in seen:
            seen.add(key)
            unique_blocks.append(block)
    
    print(f"Total events after deduplication: {len(unique_blocks)}", file=sys.stderr)

    # Write calendars
    os.makedirs("public", exist_ok=True)
    for kind, fname, title in [
        ("lap", "ekac-lap.ics", "Brookline EKAC — Lap Lanes"),
        ("shallow", "ekac-shallow.ics", "Brookline EKAC — Shallow Pool"),
        ("dive", "ekac-dive.ics", "Brookline EKAC — Dive Well"),
        (None, "ekac.ics", "Brookline EKAC — All Pools"),
    ]:
        cal = make_calendar(unique_blocks, kind if isinstance(kind, str) else None, title)  # type: ignore[arg-type]
        with open(f"public/{fname}", "wb") as f:
            f.write(cal.to_ical())

    _write_index()
    if DEBUG:
        _write_debug(all_debug, unique_blocks)

    print(
        f"Wrote feeds: ekac-lap.ics, ekac-shallow.ics, ekac-dive.ics, ekac.ics "
        f"from Google Sheets ({len(unique_blocks)} events)"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        raise
