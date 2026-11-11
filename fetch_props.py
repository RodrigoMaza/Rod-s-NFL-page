"""
Pull NFL player props from The Odds API for this week's games.

- Reads the API key from the ODDS_API_KEY environment variable (a GitHub secret).
- Only pulls games that haven't started and kick off within DAYS_AHEAD days.
- MAX_GAMES caps how many games we pull (each game costs 4 credits).
- Saves data/props.csv (one row per book / player / market / Over-Under).
"""
import csv
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

API_KEY = os.environ.get("ODDS_API_KEY")
MAX_GAMES = int(os.environ.get("MAX_GAMES", "20"))
DAYS_AHEAD = int(os.environ.get("DAYS_AHEAD", "6"))

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


# 1) Upcoming games (free) -> keep only not-started games in the window
now = datetime.now(timezone.utc)
cutoff = now + timedelta(days=DAYS_AHEAD)
events = get(f"{BASE}/events", {})
week_games = [
    e for e in events
    if now < datetime.fromisoformat(e["commence_time"].replace("Z", "+00:00")) < cutoff
][:MAX_GAMES]
print(f"Pulling props for {len(week_games)} games (est. {len(week_games) * 4} credits)\n")

# 2) Props per game
rows = []
for e in week_games:
    print(f"{e['away_team']} @ {e['home_team']}")
    data = get(
        f"{BASE}/events/{e['id']}/odds",
        {"regions": REGIONS, "markets": MARKETS, "oddsFormat": "american"},
    )
    for book in data.get("bookmakers", []):
        for market in book.get("markets", []):
            for o in market.get("outcomes", []):
                rows.append({
                    "game_id": e["id"],
                    "home_team": e["home_team"],
                    "away_team": e["away_team"],
                    "kickoff": e["commence_time"],
                    "book": book["title"],
                    "market": market["key"],
                    "player": o.get("description"),
                    "side": o.get("name"),       # Over / Under
                    "line": o.get("point"),
                    "odds": o.get("price"),
                })

# 3) Save
Path("data").mkdir(exist_ok=True)
fields = ["game_id", "home_team", "away_team", "kickoff", "book",
          "market", "player", "side", "line", "odds"]
with open("data/props.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
print(f"\nSaved {len(rows)} prop rows to data/props.csv")
