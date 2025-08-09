# bhs-pool-calendar

Scrapes the latest Brookline High School / Evelyn Kirrane Aquatics Center **Pool Schedule** PDF and publishes an
iCal feed showing **lap-lane availability** (“Lap lanes: N or N–M open”) via GitHub Actions + Pages. Zero servers.

**Calendar URL (after enabling GitHub Pages):**
https://samuelduchesne.github.io/bhs-pool-calendar/ekac.ics

## How it works
- Discovers the newest **Pool Schedule** PDF from the Aquatics page; falls back to a stable DocumentCenter ID.
- Parses Monday–Sunday columns and converts each time block with lane counts into a calendar event.
- Exports events in **UTC** so calendar apps render correctly in local time (America/New_York).
- Publishes daily at ~05:10 ET and on manual runs via GitHub Actions. Output is served by **GitHub Pages**.

## Quick deploy (phone-friendly)
1. Create repo `bhs-pool-calendar` (Public).
2. Add `.github/workflows/publish.yml` (see repo) and commit to `main`.
3. Repo → **Settings → Pages** → Source: **GitHub Actions**.
4. **Actions** tab → run “Publish EKAC lanes calendar”.
5. Subscribe to the Calendar URL above in Apple/Google Calendar.

## Local development
```bash
python -m venv .venv && source .venv/bin/activate
pip install requests pdfplumber python-dateutil icalendar beautifulsoup4
python app/build_calendar.py
open public/ekac.ics
