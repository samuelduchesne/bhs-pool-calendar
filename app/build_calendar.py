#!/usr/bin/env python3
"""Build iCal feeds for EKAC Lap, Shallow, and Dive, with optional debug artifacts.

- Finds the latest Pool Schedule PDF (Aquatics page; fallback to fixed ID).
- Splits the page into weekday columns (left-biased boundary to prevent cross-column bleed).
- Clusters text into visual rows; classifies each row into Lap / Shallow / Dive via:
    1) page-level vertical bands inferred from any section headers found anywhere on the page,
    2) falling back to per-column section markers above the row,
    3) defaulting to Lap if neither is found.
- Extracts time ranges; for Lap rows it also extracts lane counts (to the RIGHT of the time).
- Writes four feeds:
    public/ekac-lap.ics
    public/ekac-shallow.ics
    public/ekac-dive.ics
    public/ekac.ics   (combined)
- Writes public/index.html (landing page with “About/Disclaimer”).
- If DEBUG=1, also writes public/debug.html and public/debug.json.

Env:
  DEBUG=1                 -> write debug artifacts
  ROW_TOL=3.5             -> row clustering tolerance (px)
  COL_GUTTER_PX=16        -> left-biased gutter (px) near day boundaries
  COL_LEFT_BIAS_PX=6      -> tie-break toward the left column (px)
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
from typing import Iterable, Literal, Tuple

import pdfplumber
import requests
from bs4 import BeautifulSoup
from dateutil import parser as dtparse
from icalendar import Calendar, Event

AQUATICS_URL = "https://www.brooklinerec.com/150/Aquatics-Center"
FALLBACK_PDF = "https://www.brooklinerec.com/DocumentCenter/View/4404/Pool-Schedule?bidId="

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

DEBUG = os.environ.get("DEBUG", "0") == "1"


def _get_float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


ROW_TOL = _get_float_env("ROW_TOL", 3.5)
COL_GUTTER_PX = _get_float_env("COL_GUTTER_PX", 16.0)
COL_LEFT_BIAS_PX = _get_float_env("COL_LEFT_BIAS_PX", 6.0)

Kind = Literal["lap", "shallow", "dive"]


@dataclass(frozen=True)
class Block:
    start: datetime  # UTC
    end: datetime  # UTC
    kind: Kind
    label: str  # "5", "4–5" for lap; "open" for shallow/dive
    source_url: str
    page: int
    day: str
    row_text: str


SECTION_KEYS = {
    "lap": ["lap lane", "lap lanes", "lap swim", "lap"],
    "shallow": ["shallow pool", "shallow"],
    "dive": ["dive well", "diving well", "dive"],
}


def discover_latest_pdf() -> str:
    """Try to discover the latest Pool Schedule PDF; fall back to a stable ID."""
    try:
        resp = requests.get(AQUATICS_URL, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            text = (a.get_text() or "").lower()
            if ("/DocumentCenter/View/" in href) and ("pool" in text and "schedule" in text):
                return requests.compat.urljoin(AQUATICS_URL, href)
        # fallback to href-only match if the anchor text is missing
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "/DocumentCenter/View/" in href and "Pool" in href and "Schedule" in href:
                return requests.compat.urljoin(AQUATICS_URL, href)
    except Exception:
        pass
    return FALLBACK_PDF


def month_span_to_dates(header: str, now_utc: datetime) -> Tuple[datetime, datetime] | None:
    """Parse 'Pool Schedule for August 4-10' into (start_date, end_date) at midnight UTC."""
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
    """Assign tokens to weekday columns with a left-biased boundary gutter."""
    name_map = {
        "monday": "Monday",
        "mon": "Monday",
        "tuesday": "Tuesday",
        "tue": "Tuesday",
        "tues": "Tuesday",
        "wednesday": "Wednesday",
        "wed": "Wednesday",
        "thursday": "Thursday",
        "thu": "Thursday",
        "thur": "Thursday",
        "thurs": "Thursday",
        "friday": "Friday",
        "fri": "Friday",
        "saturday": "Saturday",
        "sat": "Saturday",
        "sunday": "Sunday",
        "sun": "Sunday",
    }

    headers: list[tuple[str, float]] = []
    for w in words:
        key = name_map.get((w.get("text") or "").strip().lower())
        if key:
            headers.append((key, (w["x0"] + w["x1"]) / 2.0))

    debug = {"headers_raw": headers}

    best: dict[str, float] = {}
    for name, xmid in sorted(headers, key=lambda t: t[1]):
        if name not in best:
            best[name] = xmid

    if len(best) >= 7:
        centers = [
            ("Monday", best["Monday"]),
            ("Tuesday", best["Tuesday"]),
            ("Wednesday", best["Wednesday"]),
            ("Thursday", best["Thursday"]),
            ("Friday", best["Friday"]),
            ("Saturday", best["Saturday"]),
            ("Sunday", best["Sunday"]),
        ]
        debug["mode"] = "headers"
    else:
        # Fallback: approximate equal-width centers across page
        if not words:
            minx, maxx = 0.0, 1000.0
        else:
            minx = min(w["x0"] for w in words)
            maxx = max(w["x1"] for w in words)
        step = (maxx - minx) / 7.0 if maxx > minx else 100.0
        centers = []
        for i, day in enumerate(DAYS):
            left = minx + i * step
            right = minx + (i + 1) * step
            centers.append((day, (left + right) / 2.0))
        debug["mode"] = "equal-bands"

    bounds = [(centers[i][1] + centers[i + 1][1]) / 2.0 for i in range(len(centers) - 1)]
    debug["bounds"] = bounds
    debug["centers"] = centers

    columns: dict[str, list[dict]] = {d: [] for d, _ in centers}
    for w in words:
        xmid = (w["x0"] + w["x1"]) / 2.0
        dists = [abs(xmid - c[1]) for c in centers]
        j = min(range(len(dists)), key=lambda k: dists[k])
        if j > 0:
            left_boundary = (centers[j - 1][1] + centers[j][1]) / 2.0
            if xmid - left_boundary <= COL_GUTTER_PX and (dists[j] - dists[j - 1]) <= COL_LEFT_BIAS_PX:
                j -= 1
        columns[centers[j][0]].append(w)

    for d in columns:
        columns[d].sort(key=lambda ww: (ww["top"], ww["x0"]))
    return columns, debug


def _section_markers(words: list[dict]) -> list[tuple[float, str]]:
    """Return [(y_top, kind)] for section headers within a single day column."""
    markers: list[tuple[float, str]] = []
    for w in words:
        txt = (w.get("text") or "").strip().lower()
        for kind, keys in SECTION_KEYS.items():
            if any(k in txt for k in keys):
                markers.append((w["top"], kind))  # y position
                break
    markers.sort(key=lambda t: t[0])
    return markers


def _page_section_bands(words: list[dict]) -> list[tuple[float, float, str]]:
    """Return vertical bands [(y_top, y_bottom, kind)] for lap/shallow/dive across the page.

    Strategy: find y-centers of any section header tokens anywhere on the page; sort by y and
    convert to bands by midpoints. If <2 markers found, return [] (no page bands).
    """
    hits: list[tuple[float, str]] = []
    for w in words:
        t = (w.get("text") or "").strip().lower()
        for kind, phrases in SECTION_KEYS.items():
            if any(p in t for p in phrases):
                ymid = (w["top"] + w.get("bottom", w["top"])) / 2.0
                hits.append((ymid, kind))
                break
    if not hits:
        return []

    # take the median y per kind to reduce noise
    from statistics import median

    per_kind: dict[str, list[float]] = {}
    for y, k in hits:
        per_kind.setdefault(k, []).append(y)
    centers = [(median(ys), k) for k, ys in per_kind.items()]
    if len(centers) < 2:
        return []

    centers.sort(key=lambda x: x[0])  # by y
    edges = [(centers[i][0] + centers[i + 1][0]) / 2.0 for i in range(len(centers) - 1)]

    bands: list[tuple[float, float, str]] = []
    for i, (y, kind) in enumerate(centers):
        top = -1e9 if i == 0 else edges[i - 1]
        bot = 1e9 if i == len(centers) - 1 else edges[i]
        bands.append((top, bot, kind))
    return bands


def _assign_kind(
    row: list[dict],
    bands: list[tuple[float, float, str]] | None,
    col_markers: list[tuple[float, str]] | None,
    default: str = "lap",
) -> str:
    """Prefer page-level bands; fall back to column markers; else default."""
    if not row:
        return default
    y = sum(w["top"] for w in row) / len(row)

    if bands:
        for top, bot, kind in bands:
            if top <= y <= bot:
                return kind

    if col_markers:
        above = [m for m in col_markers if m[0] <= y + 0.01]
        if above:
            return above[-1][1]

    return default


def extract_blocks_for_day(
    day_words: list[dict],
    local_date: datetime,
    page_bands: list[tuple[float, float, str]] | None,
) -> Iterable[Block]:
    """Yield Blocks for the day; Lap rows require right-side lane counts; Shallow/Dive just need time."""
    time_re = re.compile(
        r"(?P<s>\d{1,2}(?::\d{2})?\s*(?:a|p|am|pm))\s*[-–]\s*(?P<e>\d{1,2}(?::\d{2})?\s*(?:a|p|am|pm))",
        re.IGNORECASE,
    )
    lanes_right_re = re.compile(
        r"(?<!\d)(?P<n1>\d{1,2})(?:\s*[–-]\s*(?P<n2>\d{1,2}))?(?:\s*lanes?)?(?!\d)",
        re.IGNORECASE,
    )

    col_markers = _section_markers(day_words)
    for row in _rows_from_words(day_words, tol=ROW_TOL):
        row_text = " ".join((w.get("text") or "") for w in row)
        tm = time_re.search(row_text)
        if not tm:
            continue

        kind = _assign_kind(row, page_bands, col_markers, default="lap")

        sh, sm = time_token_to_24h(tm.group("s"))
        eh, em = time_token_to_24h(tm.group("e"))
        st_utc = to_utc_local_eastern(local_date, sh, sm)
        en_utc = to_utc_local_eastern(local_date, eh, em)
        if en_utc <= st_utc:
            en_utc += timedelta(days=1)

        if kind == "lap":
            post = row_text[tm.end():]
            m2 = lanes_right_re.search(post)
            if not m2:
                continue  # lap rows must include counts to the right of the time
            n1 = int(m2.group("n1"))
            n2 = m2.group("n2")
            label = f"{n1}–{int(n2)}" if n2 else f"{n1}"
        else:
            label = "open"

        yield Block(
            start=st_utc,
            end=en_utc,
            kind=kind,  # type: ignore[arg-type]
            label=label,
            source_url="",
            page=0,
            day="",
            row_text=row_text,
        )


def parse_pdf(url: str) -> tuple[list[Block], dict]:
    """Download and parse the PDF into blocks; collect debug info for inspection."""
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
            page_bands = _page_section_bands(words)

            page_dbg = {
                "page": page_index,
                "header": header,
                "week_start_utc": week_start.isoformat(),
                "column_mode": dbg_cols.get("mode"),
                "bounds": dbg_cols.get("bounds"),
                "headers_raw": dbg_cols.get("headers_raw"),
                "page_bands": page_bands,
                "days": {},
            }

            for offset, day in enumerate(DAYS):
                local_date = week_start + timedelta(days=offset)
                day_words = columns.get(day, [])
                rows_txt = [" ".join((w.get("text") or "") for w in row) for row in _rows_from_words(day_words)]
                page_dbg["days"][day] = {"row_count": len(rows_txt), "rows": rows_txt, "matches": []}

                for blk in extract_blocks_for_day(day_words, local_date, page_bands):
                    # fill metadata
                    blk = Block(
                        start=blk.start,
                        end=blk.end,
                        kind=blk.kind,
                        label=blk.label,
                        source_url=url,
                        page=page_index,
                        day=day,
                        row_text=blk.row_text,
                    )
                    blocks.append(blk)
                    page_dbg["days"][day]["matches"].append(
                        {
                            "start": blk.start.isoformat(),
                            "end": blk.end.isoformat(),
                            "kind": blk.kind,
                            "label": blk.label,
                            "row_text": blk.row_text,
                        }
                    )

            debug["pages"].append(page_dbg)

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
        ev.add("description", f"{b.day} — {b.row_text}\nSource: {b.source_url} (page {b.page})")
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
    lines.append(f"<h2>PDF: <code>{debug.get('pdf')}</code></h2>")
    lines.append(f"<p>Total parsed events: <b>{len(blocks)}</b></p>")
    kind_counts = {"lap": 0, "shallow": 0, "dive": 0}
    for b in blocks:
        kind_counts[b.kind] += 1
    lines.append(f"<p>By kind: {kind_counts}</p>")
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
        lines.append("<details><summary>Page section bands</summary><pre>")
        lines.append(json.dumps({"page_bands": page.get("page_bands")}, indent=2))
        lines.append("</pre></details>")
        for day in DAYS:
            d = page["days"].get(day, {})
            matches = d.get("matches", [])
            cls = "ok" if matches else "bad"
            lines.append(f"<details open><summary class='{cls}'>{day}: {len(matches)} matches</summary>")
            rows = d.get("rows", [])
            lines.append("<pre>Rows:\n" + ("\n".join(rows) if rows else "(none)") + "</pre>")
            if matches:
                lines.append("<pre>Matches:\n" + json.dumps(matches, indent=2) + "</pre>")
            lines.append("</details>")
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
    Unofficial, community-built calendars generated from the weekly Pool Schedule PDF published by the
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
  <p>Shows lane availability (e.g., “Lap lanes: 4–5 open”).</p>
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
    pdf_url = discover_latest_pdf()
    blocks, debug = parse_pdf(pdf_url)

    # Write calendars
    os.makedirs("public", exist_ok=True)
    for kind, fname, title in [
        ("lap", "ekac-lap.ics", "Brookline EKAC — Lap Lanes"),
        ("shallow", "ekac-shallow.ics", "Brookline EKAC — Shallow Pool"),
        ("dive", "ekac-dive.ics", "Brookline EKAC — Dive Well"),
        (None, "ekac.ics", "Brookline EKAC — All Pools"),
    ]:
        cal = make_calendar(blocks, kind if isinstance(kind, str) else None, title)  # type: ignore[arg-type]
        with open(f"public/{fname}", "wb") as f:
            f.write(cal.to_ical())

    _write_index()
    if DEBUG:
        _write_debug(debug, blocks)

    print(
        "Wrote feeds: ekac-lap.ics, ekac-shallow.ics, ekac-dive.ics, ekac.ics "
        f"from {debug.get('pdf')}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
