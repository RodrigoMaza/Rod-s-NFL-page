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
from scipy.stats import gamma, poisson

MIN_EDGE = 0.04          # flag only if our (market-blended) probability beats Vegas by 4+ points
MAX_EDGE = 0.12          # bigger than this usually means news we don't know (injury, role change)
MODEL_WEIGHT = 0.5       # Vegas is sharp: final probability = 50% our model + 50% Vegas
TOP_N = 10               # show at most the 10 best picks per week
MAX_PER_TEAM = 2         # picks on the same team are usually the same bet (e.g. "offense struggles")
MIN_LINE = {"player_pass_yds": 100, "player_rush_yds": 10,   # skip tiny lines (QB kneel-downs,
            "player_reception_yds": 10, "player_receptions": 1.5}  # backup RBs) - math breaks there
ROLE_CHANGE = 0.40       # skip if last-4 usage differs 40%+ from season (Vegas knows why, we don't)
MIN_SNAPS = 0.40         # skip part-time players (non-QB) under 40% of snaps lately
# skip if projection / line is outside this range: a gap that big means Vegas is reacting to news
# (yardage averages run ~20% above typical games, so yards get a little more room on the high side)
RATIO_RANGE = {"player_pass_yds": (0.75, 1.35), "player_rush_yds": (0.6, 1.7),
               "player_reception_yds": (0.6, 1.7), "player_receptions": (0.67, 1.5)}
RECENT_WEIGHT = 0.6      # 60% last 4 games, 40% season
MATCHUP_WEIGHT = 0.5     # use half of the matchup effect (9 games of defense data is noisy)
MIN_GAMES = 6
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


def p_over(line, mean, sd, market):
    """Chance of going over.
    Yards: right-skewed (gamma) curve - most games land a bit below the average.
    Receptions: whole numbers, so a Poisson count model."""
    if market == "player_receptions":
        raw = poisson.sf(int(line), mean)          # P(catches > line), e.g. line 4.5 -> 5+
    else:
        sd = sd * SD_INFLATE
        raw = gamma.sf(line, a=(mean / sd) ** 2, scale=sd ** 2 / mean)
    return 0.5 + SHRINK * (raw - 0.5)


players = pd.read_csv("data/players.csv")
qbs = pd.read_csv("data/qbs.csv")
qb_unsure = set(qbs[qbs.qb_uncertain].team)
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
            "snaps_last4": p.offense_pct_last4,
            "role_change": abs(recent / season - 1) if pd.notna(recent) and season > 0 else 0,
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
m = m[m.line >= m.market.map(MIN_LINE)]
m["p_over"] = [p_over(l, mu, s, mk) for l, mu, s, mk in zip(m.line, m.proj, m.sd, m.market)]
# de-bias: on average Vegas is right, so remove any overall Over/Under lean per prop type
over = m.side == "Over"
lean = (m[over].p_over - m[over].fair_p).groupby(m[over].market).mean()
m["p_over"] = m.p_over - m.market.map(lean).fillna(0)
print("Removed model lean vs Vegas (P over):", {LABEL[k]: round(v, 3) for k, v in lean.items()})
raw_p = m.p_over.where(over, 1 - m.p_over)
m["model_p"] = MODEL_WEIGHT * raw_p + (1 - MODEL_WEIGHT) * m.fair_p
m["edge"] = m.model_p - m.fair_p
m["ev_per_100"] = 100 * (m.model_p * m.odds.apply(payout) - (1 - m.model_p))
m["prop"] = m.market.map(LABEL)

cols = ["player_display_name", "position", "team", "next_opp", "prop", "side", "line",
        "book", "odds", "proj", "model_p", "fair_p", "edge", "ev_per_100", "games",
        "snaps_last4", "role_change", "kickoff"]
edges = m[cols].rename(columns={"player_display_name": "player"})
edges.round(3).to_csv("data/edges.csv", index=False)

# ---- 5) Picks: best book for each player/prop/side, flagged edges only
best = edges.sort_values("edge", ascending=False).drop_duplicates(["player", "prop", "side"])
lo = best.prop.map({LABEL[k]: v[0] for k, v in RATIO_RANGE.items()})
hi = best.prop.map({LABEL[k]: v[1] for k, v in RATIO_RANGE.items()})
ratio = best.proj / best.line
ok = ((best.edge >= MIN_EDGE) & ratio.between(lo, hi) & (best.edge <= MAX_EDGE) & (best.games >= MIN_GAMES)
      & (best.role_change < ROLE_CHANGE)
      & ((best.position == "QB") | (best.snaps_last4 >= MIN_SNAPS)))
# QB change or injury: the passing game (QB yards, all receiving) is unpredictable -> skip
passing = best.prop.isin(["Pass yds", "Rec yds", "Receptions"])
ok = ok & ~(passing & best.team.isin(qb_unsure))
picks = best[ok].copy()
# one pick per player (yards and receptions for the same player are basically the same bet)
picks = picks.sort_values("edge", ascending=False).drop_duplicates("player")
picks = picks.groupby("team").head(MAX_PER_TEAM).head(TOP_N)
picks["note"] = ""
picks.round(3).to_csv("data/picks.csv", index=False)

n_raw = (best.edge >= MIN_EDGE).sum()
print(f"\nPriced {len(best)} props. {n_raw} cleared {MIN_EDGE:.0%}; {len(picks)} kept after filters "
      f"(QB settled, games >= {MIN_GAMES}, steady role, regular snaps, projection not wildly off the line, 1 per player).\n")
show = picks.head(15).copy()
for c in ["model_p", "fair_p", "edge"]:
    show[c] = (show[c] * 100).round(0).astype(int).astype(str) + "%"
show["proj"] = show.proj.round(1)
print(show[["player", "team", "prop", "side", "line", "book", "odds",
            "proj", "model_p", "fair_p", "edge", "note"]].to_string(index=False))
