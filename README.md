# bhs-pool-calendar

Generates iCal feeds for the Evelyn Kirrane Aquatics Center (Brookline High School) pool schedule by parsing the
weekly PDF and publishing calendars via GitHub Actions + Pages.

## About & Disclaimer

> **Unofficial project.** This repository is maintained by the community and is not affiliated with the Town of
> Brookline. The feeds are generated automatically by scraping the public PDF. **No guarantees**: parsing can be wrong,
> stale, or incomplete. **Always confirm** with the facility’s official schedule:
> https://www.brooklinerec.com/150/Aquatics-Center  
> This is just a fun project using ChatGPT 5.

---

## Feeds

Landing page (choose your calendar):  
https://samuelduchesne.github.io/bhs-pool-calendar/

Direct feeds:
- **Lap lanes:** https://samuelduchesne.github.io/bhs-pool-calendar/ekac-lap.ics
- **Shallow pool:** https://samuelduchesne.github.io/bhs-pool-calendar/ekac-shallow.ics
- **Dive well:** https://samuelduchesne.github.io/bhs-pool-calendar/ekac-dive.ics
- **All-in-one:** https://samuelduchesne.github.io/bhs-pool-calendar/ekac.ics

> Subscribe in Apple/Google Calendar by adding the `.ics` URL.

## How it works

- Finds the latest “Pool Schedule” PDF on the Aquatics page (with a stable fallback).
- Parses weekday columns using page geometry; clusters text into rows to extract time ranges.
- Builds four iCal feeds (Lap / Shallow / Dive / All) and publishes them to GitHub Pages on a daily cron
  and on manual runs.

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app/build_calendar.py
open public/index.html
```

Environment toggles (used by the workflow):

- `DEBUG=1` → publish debug.html / debug.json for troubleshooting
- `ROW_TOL` (default 3.5) → row clustering tolerance (pixels)
- `COL_GUTTER_PX` (default 16) → left-bias gutter to prevent Sat/Sun column bleed
- `COL_LEFT_BIAS_PX` (default 6) → tie-break toward the left column

### Contributing

Issues and PRs are welcome—especially if the PDF layout changes and parsing needs tweaks.
If you report a parsing problem, please include:
	•	Date/day and the affected row text from debug.html
	•	What the expected events should be

### License

MIT. See [LICENSE](LICENSE).
