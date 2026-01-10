# Google Sheets Format

The pool schedule parser expects a Google Sheets document with the following structure:

## Multiple Sheets Support

The parser automatically discovers and processes **all sheets/tabs** in the Google Sheets document. Each sheet typically represents a different week of pool schedules. When the pool administration adds new tabs for future weeks, they will be automatically included in the generated calendars.

## Expected Format

```csv
Pool Schedule for [Month] [Start Day]-[End Day]

,Monday,Tuesday,Wednesday,Thursday,Friday,Saturday,Sunday
[Time Range],[Activity],[Activity],[Activity],[Activity],[Activity],[Activity],[Activity]
[Time Range],[Activity],[Activity],[Activity],[Activity],[Activity],[Activity],[Activity]
...
```

## Example

```csv
Pool Schedule for January 13-19, 2026

,Monday,Tuesday,Wednesday,Thursday,Friday,Saturday,Sunday
6am - 8am,Lap lanes 4-5,Lap lanes 4-5,Lap lanes 4-5,Lap lanes 4-5,Lap lanes 4-5,Lap lanes 3-4,Lap lanes 3-4
8am - 10am,Shallow pool open,Shallow pool open,Shallow pool open,Lap lanes 5,Lap lanes 5,Dive well open,Dive well open
10am - 12pm,Lap lanes 3,Lap lanes 3,Lap lanes 3,Shallow pool open,Shallow pool open,Shallow pool open,Shallow pool open
```

## Activity Classification

The parser automatically classifies activities into three types:

1. **Lap lanes**: Text containing "lap lane", "lap lanes", "lap swim", or "lap"
   - Must include lane counts (e.g., "4-5", "5") for the event to be included in the lap feed
   - Example: "Lap lanes 4-5", "Lap 3 lanes"

2. **Shallow pool**: Text containing "shallow pool" or "shallow"
   - Example: "Shallow pool open", "Shallow"

3. **Dive well**: Text containing "dive well", "diving well", or "dive"
   - Example: "Dive well open", "Dive"

## Time Format

The parser supports multiple time formats:
- `7am - 9am`
- `7:30am - 9:00pm`
- `7a - 9p`
- `7:15 am - 9:30 pm`

## Header Detection

The parser automatically detects:
1. The week range from any row containing "Pool Schedule for [Month] [Days]"
2. The day columns by finding cells with day names (Monday, Mon, Tuesday, Tue, etc.)
3. Data rows that contain time ranges

## Configuration

The Google Sheets URL is hardcoded in the script but can be overridden with the `SHEETS_URL` environment variable:

```bash
export SHEETS_URL="https://docs.google.com/spreadsheets/d/YOUR_SHEET_ID/export?format=csv"
python app/build_calendar.py
```

## Debugging

Set `DEBUG=1` to generate `debug.html` and `debug.json` files in the `public/` directory:

```bash
DEBUG=1 python app/build_calendar.py
```

These files show:
- Parsed rows from the CSV
- Week start date
- Day column indices
- Generated events for each pool type
