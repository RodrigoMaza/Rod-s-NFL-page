"""
Step 3: the prop model.

Reads:  data/players.csv (from fetch_stats.py)  +  data/props.csv (from fetch_props.py)
Writes: data/edges.csv  - every prop we priced, with our probability vs Vegas
        data/picks.csv  - only the flagged edges, best book per prop

How it works, per player and prop:
  1. Projection = recent form (60% last 4 games + 40% season) x matchup factor
  2. Spread     = how much this player usually swings game to game
  3. P(over)    = chance he goes over the line, using a skewed (gamma) curve,
                  then pulled halfway toward 50/50 (backtest showed the raw model was overconfident)
  4. Vegas      = the odds turned into a probability, with the bookmaker's cut removed
  5. Edge       = our probability - Vegas probability. Flag if >= MIN_EDGE.
"""
import pandas as pd
from scipy.stats import gamma

MIN_EDGE = 0.05          # flag only if we're 5+ percentage points above Vegas
MAX_EDGE = 0.25          # bigger than this usually means news we don't know (injury, role change)
RECENT_WEIGHT = 0.6      # 60% last 4 games, 40% season
MATCHUP_WEIGHT = 0.5     # use half of the matchup effect (9 games of defense data is noisy)
MIN_GAMES = 4
SD_INFLATE = 1.1         # backtest (weeks 5-9): slightly wider spread fits real outcomes better
SHRINK = 0.5             # backtest: raw model was overconfident -> keep half its distance from 50/50

# Odds API market -> (stat column in players.csv, matchup factor column)
MARKETS = {
    "player_pass_yds":      ("passing_yards",   "passing_yards_factor"),
    "player_rush_yds":      ("rushing_yards",   "rushing_yards_factor"),
    "player_reception_yds": ("receiving_yards", "receiving_yards_factor"),
    "player_receptions":    ("receptions",      "receptions_factor"),
}
LABEL = {"player_pass_yds": "Pass yds", "player_rush_yds": "Rush yds",
         "player_reception_yds": "Rec yds", "player_receptions": "Receptions"}

TEAMS = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Los Angeles Rams": "LA", "Los Angeles Chargers": "LAC",
    "Las Vegas Raiders": "LV", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "Seattle Seahawks": "SEA", "San Francisco 49ers": "SF", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}


def clean_name(s):
    return (s.str.lower()
             .str.replace(r"\b(jr|sr|ii|iii|iv|v)\b\.?", "", regex=True)
             .str.replace(r"[^a-z]", "", regex=True))


def implied(odds):
    """American odds -> probability (includes the bookmaker's cut)."""
    return -odds / (-odds + 100) if odds < 0 else 100 / (odds + 100)


def payout(odds):
    """Profit per $1 staked."""
    return odds / 100 if odds > 0 else 100 / -odds


def p_over(line, mean, sd):
    """Chance of going over, using a right-skewed (gamma) curve: most games land a bit
    below the average, with occasional big games pulling the average up."""
    sd = sd * SD_INFLATE
    raw = gamma.sf(line, a=(mean / sd) ** 2, scale=sd ** 2 / mean)
    return 0.5 + SHRINK * (raw - 0.5)


players = pd.read_csv("data/players.csv")
props = pd.read_csv("data/props.csv").dropna(subset=["player", "line", "odds"])
props = props[props.market.isin(MARKETS)]
print(f"Loaded {len(players)} players and {len(props)} prop rows")

# ---- 1) Projection + spread, per player and market
league_cv = {}
for mkt, (col, _) in MARKETS.items():
    q = players[(players.games >= MIN_GAMES) & (players[col] > 5 if col != "receptions" else players[col] > 1.5)]
    league_cv[mkt] = (q[f"{col}_sd"] / q[col]).median()

proj_rows = []
for _, p in players.iterrows():
    for mkt, (col, fcol) in MARKETS.items():
        season, recent = p[col], p[f"{col}_last4"]
        base = RECENT_WEIGHT * recent + (1 - RECENT_WEIGHT) * season if pd.notna(recent) else season
        factor = p[fcol] if pd.notna(p[fcol]) else 1.0
        factor = min(max(1 + MATCHUP_WEIGHT * (factor - 1), 0.85), 1.15)
        proj = base * factor
        # blend player's own swing with a league-typical swing, so small samples don't mislead
        own_sd = p[f"{col}_sd"] if pd.notna(p[f"{col}_sd"]) else league_cv[mkt] * proj
        sd = 0.5 * own_sd + 0.5 * league_cv[mkt] * proj
        sd = max(sd, 0.25 * proj, 1.0 if col == "receptions" else 8.0)
        proj_rows.append({
            "key": None, "player_display_name": p.player_display_name, "team": p.team,
            "position": p.position, "games": p.games, "next_opp": p.next_opp,
            "market": mkt, "proj": proj, "sd": sd,
        })
proj = pd.DataFrame(proj_rows)
proj["key"] = clean_name(proj.player_display_name)

# ---- 2) Vegas: pair Over/Under at each book, remove the bookmaker's cut
props["key"] = clean_name(props.player)
props["home"] = props.home_team.map(TEAMS)
props["away"] = props.away_team.map(TEAMS)
props["imp"] = props.odds.apply(implied)
pair = props.groupby(["book", "key", "market", "line"]).imp.transform("sum")
n_sides = props.groupby(["book", "key", "market", "line"]).side.transform("nunique")
props = props[n_sides == 2].copy()
props["fair_p"] = props.imp / pair[n_sides == 2]

# ---- 3) Match props to our projections (same name, team playing in this game)
m = props.merge(proj, on=["key", "market"], how="left")
m = m[m.team.isna() | (m.team == m.home) | (m.team == m.away)]
unmatched = m[m.proj.isna()].player.unique()
m = m.dropna(subset=["proj"])
print(f"Matched {m.player.nunique()} players to stats; {len(unmatched)} not matched")
if len(unmatched):
    print("  Not matched (rookies/new players or name differences):", ", ".join(sorted(unmatched)[:15]))

# ---- 4) Our probability, edge and expected value
m["p_over"] = [p_over(l, mu, s) for l, mu, s in zip(m.line, m.proj, m.sd)]
m["model_p"] = m.p_over.where(m.side == "Over", 1 - m.p_over)
m["edge"] = m.model_p - m.fair_p
m["ev_per_100"] = 100 * (m.model_p * m.odds.apply(payout) - (1 - m.model_p))
m["prop"] = m.market.map(LABEL)

cols = ["player_display_name", "position", "team", "next_opp", "prop", "side", "line",
        "book", "odds", "proj", "model_p", "fair_p", "edge", "ev_per_100", "games", "kickoff"]
edges = m[cols].rename(columns={"player_display_name": "player"})
edges.round(3).to_csv("data/edges.csv", index=False)

# ---- 5) Picks: best book for each player/prop/side, flagged edges only
best = edges.sort_values("edge", ascending=False).drop_duplicates(["player", "prop", "side"])
picks = best[(best.edge >= MIN_EDGE) & (best.games >= MIN_GAMES)].copy()
picks["note"] = picks.edge.apply(lambda e: "check news" if e > MAX_EDGE else "")
picks = picks.sort_values("edge", ascending=False)
picks.round(3).to_csv("data/picks.csv", index=False)

print(f"\nPriced {len(best)} props. Flagged {len(picks)} with edge >= {MIN_EDGE:.0%}.\n")
show = picks.head(15).copy()
for c in ["model_p", "fair_p", "edge"]:
    show[c] = (show[c] * 100).round(0).astype(int).astype(str) + "%"
show["proj"] = show.proj.round(1)
print(show[["player", "team", "prop", "side", "line", "book", "odds",
            "proj", "model_p", "fair_p", "edge", "note"]].to_string(index=False))
