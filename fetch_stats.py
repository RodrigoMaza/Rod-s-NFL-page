"""
Step 2: pull player stats from nflverse and build a per-player summary
for the upcoming week. No API key needed (nflverse is free/open, CC BY 4.0).

Outputs:
  data/players.csv  - one row per QB/RB/WR/TE: volume, efficiency, recent form,
                      next opponent and matchup factor
  data/defense.csv  - yards each defense allows per game, by position
"""
import sys
from datetime import date
from pathlib import Path

import pandas as pd

SEASON = 2026
RECENT_GAMES = 4          # "recent form" window
BASE = "https://github.com/nflverse/nflverse-data/releases/download"
POSITIONS = ["QB", "RB", "WR", "TE"]


def load(path):
    print(f"Downloading {path} ...")
    return pd.read_csv(f"{BASE}/{path}", low_memory=False)


stats = load(f"stats_player/stats_player_week_{SEASON}.csv")
snaps = load(f"snap_counts/snap_counts_{SEASON}.csv")
games = load("schedules/games.csv")

stats = stats[(stats.season_type == "REG") & (stats.position.isin(POSITIONS))]
games = games[(games.season == SEASON) & (games.game_type == "REG")]

last_week = int(stats.week.max())
upcoming = games[games.result.isna()]
if upcoming.empty:
    sys.exit("No upcoming games found (season over?).")
next_week = int(upcoming.week.min())
print(f"Latest week with stats: {last_week} | Next week to project: {next_week}")

# ---- 1) Add snap share to each player-week (match on name + team + week)
def clean_name(s):
    # "Luther Burden III" / "Luther Burden" -> "lutherburden"
    return (s.str.lower()
             .str.replace(r"\b(jr|sr|ii|iii|iv|v)\b\.?", "", regex=True)
             .str.replace(r"[^a-z]", "", regex=True))

stats["key"] = clean_name(stats.player_display_name)
snaps["key"] = clean_name(snaps.player)
stats = stats.merge(
    snaps[["key", "team", "week", "offense_pct"]],
    on=["key", "team", "week"], how="left",
)
matched = stats.offense_pct.notna().mean()
print(f"Snap % matched for {matched:.0%} of player-games")

# ---- 2) Defense: yards allowed per game, by position
allowed = (
    stats.groupby(["opponent_team", "position", "week"])
    [["passing_yards", "rushing_yards", "receiving_yards", "receptions"]]
    .sum().groupby(["opponent_team", "position"]).mean().reset_index()
    .rename(columns={"opponent_team": "defense"})
)
# matchup factor = this defense / league average (1.10 = allows 10% more)
for col in ["passing_yards", "rushing_yards", "receiving_yards", "receptions"]:
    league_avg = allowed.groupby("position")[col].transform("mean")
    allowed[f"{col}_factor"] = (allowed[col] / league_avg).round(2)

# ---- 3) Player summary: season + last-N averages
COLS = ["attempts", "passing_yards", "carries", "rushing_yards",
        "targets", "receptions", "receiving_yards", "target_share", "offense_pct"]

stats = stats.sort_values("week")
season_avg = stats.groupby(["player_id", "player_display_name", "position", "team"])[COLS].mean()
season_avg["games"] = stats.groupby(["player_id", "player_display_name", "position", "team"]).size()
recent = (stats.groupby("player_id").tail(RECENT_GAMES)
          .groupby("player_id")[COLS].mean().add_suffix("_last4"))

# game-to-game spread (std dev) -> how "swingy" each player is
SD_COLS = ["passing_yards", "rushing_yards", "receiving_yards", "receptions"]
spread = stats.groupby("player_id")[SD_COLS].std().add_suffix("_sd")

players = (season_avg.reset_index()
           .merge(recent, on="player_id", how="left")
           .merge(spread, on="player_id", how="left"))

# keep players who are actually involved (drop deep backups)
players = players[(players.games >= 2) & (
    (players.attempts >= 10) | (players.carries >= 4) | (players.targets >= 2.5))]

# ---- 4) Next opponent + matchup factor
wk = games[games.week == next_week]
opp = pd.concat([
    wk[["home_team", "away_team"]].rename(columns={"home_team": "team", "away_team": "next_opp"}),
    wk[["away_team", "home_team"]].rename(columns={"away_team": "team", "home_team": "next_opp"}),
])
players = players.merge(opp, on="team", how="left")   # NaN next_opp = bye week
players = players.merge(
    allowed[["defense", "position", "passing_yards_factor", "rushing_yards_factor",
             "receiving_yards_factor", "receptions_factor"]],
    left_on=["next_opp", "position"], right_on=["defense", "position"], how="left",
).drop(columns="defense")

players = players.round(2).sort_values(["position", "receiving_yards_last4"], ascending=[True, False])
players.insert(0, "week", next_week)

# ---- 5) Save + print a sample
Path("data").mkdir(exist_ok=True)
players.to_csv("data/players.csv", index=False)
allowed.round(1).to_csv("data/defense.csv", index=False)

print(f"\nSaved {len(players)} players and {len(allowed)} defense rows.")
show = ["player_display_name", "team", "next_opp", "games", "targets_last4",
        "receiving_yards", "receiving_yards_last4", "offense_pct_last4", "receiving_yards_factor"]
print("\nTop WRs by receiving yards, last 4 games:")
print(players[players.position == "WR"][show].head(10).to_string(index=False))
