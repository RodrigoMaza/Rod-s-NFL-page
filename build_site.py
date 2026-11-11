"""
Step 4: build the web page.

Reads:  data/picks.csv, data/edges.csv (from model.py) + this week's schedule (nflverse)
Writes: docs/index.html            - the page (GitHub Pages serves the docs/ folder)
        history/week_XX_picks.csv  - a copy of this week's picks, for the track record
"""
import json
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

SEASON = 2026
LOCAL = ZoneInfo("America/Mexico_City")
EASTERN = ZoneInfo("America/New_York")      # nflverse game times are US Eastern

NAMES = {
    "ARI": "Cardinals", "ATL": "Falcons", "BAL": "Ravens", "BUF": "Bills", "CAR": "Panthers",
    "CHI": "Bears", "CIN": "Bengals", "CLE": "Browns", "DAL": "Cowboys", "DEN": "Broncos",
    "DET": "Lions", "GB": "Packers", "HOU": "Texans", "IND": "Colts", "JAX": "Jaguars",
    "KC": "Chiefs", "LA": "Rams", "LAC": "Chargers", "LV": "Raiders", "MIA": "Dolphins",
    "MIN": "Vikings", "NE": "Patriots", "NO": "Saints", "NYG": "Giants", "NYJ": "Jets",
    "PHI": "Eagles", "PIT": "Steelers", "SEA": "Seahawks", "SF": "49ers", "TB": "Buccaneers",
    "TEN": "Titans", "WAS": "Commanders",
}
ROOF = {"dome": "Dome", "closed": "Roof closed", "open": "Roof open", "outdoors": "Outdoors"}


def implied(o):
    return -o / (-o + 100) if o < 0 else 100 / (o + 100)


def fmt_time(dt):
    return dt.strftime("%-I:%M %p").lower().replace(":00", "")


players = pd.read_csv("data/players.csv")
week = int(players.week.iloc[0])
picks = pd.read_csv("data/picks.csv")
edges = pd.read_csv("data/edges.csv")
priced = edges.drop_duplicates(["player", "prop"]).shape[0]
qbs = pd.read_csv("data/qbs.csv").set_index("team")

# ---- Games this week
games = pd.read_csv("https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv")
games = games[(games.season == SEASON) & (games.week == week) & (games.game_type == "REG")]

game_rows = []
for _, g in games.iterrows():
    kick = datetime.strptime(f"{g.gameday} {g.gametime}", "%Y-%m-%d %H:%M").replace(tzinfo=EASTERN).astimezone(LOCAL)
    if pd.notna(g.home_moneyline) and pd.notna(g.away_moneyline):
        ph, pa = implied(g.home_moneyline), implied(g.away_moneyline)
        p_home = ph / (ph + pa)
    else:
        p_home = 0.5
    s = g.spread_line          # nflverse: positive = home team favored by that many points
    if pd.isna(s) or s == 0:
        spread_txt = "Pick'em"
    else:
        fav = g.home_team if s > 0 else g.away_team
        spread_txt = f"{NAMES[fav]} by {abs(s):g}"
    game_rows.append({
        "sort": kick.isoformat(),
        "day": kick.strftime("%Y-%m-%d"),
        "day_long": kick.strftime("%A, %B %-d"),
        "time": fmt_time(kick),
        "away": g.away_team, "home": g.home_team,
        "away_name": NAMES[g.away_team], "home_name": NAMES[g.home_team],
        "away_qb": qbs.qb_shown.get(g.away_team, ""),
        "home_qb": qbs.qb_shown.get(g.home_team, ""),
        "p_home": round(p_home, 3), "p_away": round(1 - p_home, 3),
        "spread_txt": spread_txt,
        "total": f"{g.total_line:g}" if pd.notna(g.total_line) else "n/a",
        "roof": ROOF.get(g.roof, "") if pd.notna(g.roof) else "",
    })
game_rows.sort(key=lambda r: r["sort"])

# ---- Picks
UNIT = {"Pass yds": "yds", "Rush yds": "yds", "Rec yds": "yds", "Receptions": "catches"}
pick_rows = []
for _, p in picks.iterrows():
    kick = datetime.fromisoformat(str(p.kickoff).replace("Z", "+00:00")).astimezone(LOCAL)
    pick_rows.append({
        "player": p.player, "team": p.team, "opp": p.next_opp, "position": p.position,
        "prop": p.prop, "side": p.side, "line": f"{p.line:g}",
        "book": p.book, "odds": int(p.odds),
        "proj_txt": (f"{p.proj:.1f}" if p.prop == "Receptions" else f"{p.proj:.0f}") + f" {UNIT[p.prop]}",
        "model_p": round(float(p.model_p), 3), "fair_p": round(float(p.fair_p), 3),
        "edge": round(float(p.edge), 3),
        "day": kick.strftime("%a"),
    })

now = datetime.now(LOCAL)
data = {
    "week": week,
    "updated": now.strftime("%A, %B %-d at ") + fmt_time(now),
    "priced": int(priced),
    "picks": pick_rows,
    "games": game_rows,
}

# ---- Write page + history
html = Path("template.html").read_text()
html = html.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False))
Path("docs").mkdir(exist_ok=True)
Path("docs/index.html").write_text(html)
Path("docs/.nojekyll").write_text("")          # tells GitHub Pages to serve the file as-is

Path("history").mkdir(exist_ok=True)
shutil.copy("data/picks.csv", f"history/week_{week:02d}_picks.csv")

print(f"Built docs/index.html: week {week}, {len(pick_rows)} picks, {len(game_rows)} games")
