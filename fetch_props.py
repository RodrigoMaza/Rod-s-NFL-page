"""
Step 1 test: pull NFL player props from The Odds API and print them.

- Reads the API key from the ODDS_API_KEY environment variable (a GitHub secret).
- MAX_GAMES limits how many games we pull props for (default 1, to save credits).
- Saves results to data/props.json and data/props.csv.
"""
import csv
import json
import os
import sys
from pathlib import Path

import requests

API_KEY = os.environ.get("ODDS_API_KEY")
MAX_GAMES = int(os.environ.get("MAX_GAMES", "1"))

BASE = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl"
MARKETS = "player_pass_yds,player_rush_yds,player_reception_yds,player_receptions"
REGIONS = "us"

if not API_KEY:
    sys.exit("ERROR: ODDS_API_KEY is not set. Add it as a GitHub secret.")


def get(url, params):
    r = requests.get(url, params={"apiKey": API_KEY, **params}, timeout=30)
    if r.status_code != 200:
        sys.exit(f"ERROR {r.status_code}: {r.text}")
    used = r.headers.get("x-requests-used")
    remaining = r.headers.get("x-requests-remaining")
    if used is not None:
        print(f"  [credits] used: {used} | remaining: {remaining}")
    return r.json()


# 1) List upcoming games (this endpoint doesn't use credits)
print("Fetching upcoming NFL games...")
events = get(f"{BASE}/events", {})
print(f"Found {len(events)} upcoming games.\n")
for e in events[:16]:
    print(f"  {e['commence_time']}  {e['away_team']} @ {e['home_team']}")

# 2) Pull props for the first MAX_GAMES games
rows = []
raw = []
for e in events[:MAX_GAMES]:
    print(f"\nProps for {e['away_team']} @ {e['home_team']}:")
    data = get(
        f"{BASE}/events/{e['id']}/odds",
        {"regions": REGIONS, "markets": MARKETS, "oddsFormat": "american"},
    )
    raw.append(data)
    for book in data.get("bookmakers", []):
        for market in book.get("markets", []):
            for o in market.get("outcomes", []):
                rows.append({
                    "game": f"{e['away_team']} @ {e['home_team']}",
                    "kickoff": e["commence_time"],
                    "book": book["title"],
                    "market": market["key"],
                    "player": o.get("description"),
                    "side": o.get("name"),       # Over / Under
                    "line": o.get("point"),
                    "odds": o.get("price"),
                })

# 3) Save + print a sample
Path("data").mkdir(exist_ok=True)
Path("data/props.json").write_text(json.dumps(raw, indent=2))
if rows:
    with open("data/props.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

print(f"\nTotal prop rows: {len(rows)}")
for r in rows[:20]:
    print(f"  {r['book']:<14} {r['market']:<22} {str(r['player']):<24} "
          f"{r['side']:<5} {r['line']:>6}  {r['odds']}")
if not rows:
    print("  No props posted yet for this game (normal early in the week).")
