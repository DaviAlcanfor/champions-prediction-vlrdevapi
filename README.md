<div align="center">

# 🎯 VCT Champions Prediction

### Predicting the winner of Valorant Champions 2026 using machine learning

![Status](https://img.shields.io/badge/status-in%20development-orange)
![Python](https://img.shields.io/badge/python-3.13-blue)
![License](https://img.shields.io/badge/license-MIT-green)

</div>

---

## Objective

Train a machine learning model on historical VCT match data to predict **who will win Valorant Champions 2026**.

The model outputs win probabilities for each matchup, which are fed into a **Monte Carlo simulation** (10,000 iterations) of the full tournament bracket — producing a championship probability estimate for every competing team.

---

## How It Works

### Pipeline

```
Kaggle Datasets ──► Normalizer ──► Match Results ────────────────────────┐
                                                                         ▼
vlrdevapi ──────► Team Placements ──► Initial Elo ──► Feature Table ──► Model ──► Monte Carlo ──► Winner Probability
                                                                         ▲
                                     Dynamic Elo (per match) ────────────┘
```

---

## Feature Engineering

### Elo Rating

Each team has a dynamic Elo rating that updates after every match using the standard formula:

```
new_elo  = elo + K × event_weight × (result - expected)
expected = 1 / (1 + 10 ^ ((elo_opponent - elo_team) / 400))
```

Where `result = 1` if the team won, `0` if they lost. The further apart the Elo ratings, the less the winner gains — and the more the loser loses.

**K factor by stage:**

| Stage | K |
|---|---|
| Group Stage / Swiss Stage | 20 |
| Main Event | 24 |
| Playoffs | 32 |

**Event weight by tournament:**

| Tournament | Weight |
|---|---|
| Valorant Champions | 1.5 |
| Masters | 1.2 |
| Regional (Kickoff, Stage 1, Stage 2) | 0.7 |

---

### Initial Elo

Each team's starting Elo is calculated from their historical placements across VCT events, using only placements from years before the training period (so no result from the period being predicted leaks in), weighted by two factors:

| Factor | Logic |
|---|---|
| Event importance | Champions > Masters > Regional |
| Time decay | Recent results matter more (`half_life = 2 years`) |

```
weighted_score = event_weight × decay / place
initial_elo   = sum(weighted_scores) × 100
```

**Season regression:** at the start of each new season, every rating is pulled 25% of the way back toward the mean (`SEASON_REGRESSION` in `constants.py`), so last season's dominance doesn't carry over in full.

---

### Player Stats

| Feature | Description |
|---|---|
| ACS / KAST / ADR | Aggregated per team over a 150-day window before each match |
| Roster stability | Days with stable roster, new players in last 30 days *(in progress)* |
| Playstyle | Aggressiveness, carry dependency, attack preference *(in progress)* |

---

### Inter-regional Factor *(in progress)*

Cross-regional matchups (e.g. Americas vs Pacific) are flagged as a feature. Teams from different regions often have distinct playstyles, which can affect performance in international events.

---

## Model

| Component | Details |
|---|---|
| Algorithm | XGBoost |
| Validation | Walk-forward by tournament — no data leakage |
| Tuning | Optuna hyperparameter optimization (50 trials) |
| Sample weights | `weight = (time_weight ^ alpha) × (event_weight ^ (1 - alpha))` |
| Output | Win probability per matchup → Monte Carlo simulation |

### Sample Weights

Every training match is weighted by two factors balanced by `alpha` (Optuna-tuned):

```
weight = (time_weight ^ alpha) × (event_weight ^ (1 - alpha))

time_weight  = exp(-days_ago / decay_days)
event_weight = 1.5 (Champions) | 1.2 (Masters) | 0.7 (Regional)
```

A match from 2 years ago at a regional event contributes much less to training than a recent Champions match.

---

## Monte Carlo Simulation

The simulation follows the Champions 2026 format:

- **Group stage:** 4 GSL groups of 4 teams (double elimination). Top 2 of each group advance.
- **Playoffs:** 8-team double elimination bracket, from Upper Round 1 to the Grand Final.

It works as follows:

1. For each of 10,000 iterations, simulate the full tournament bracket from start to finish
2. Each matchup is decided probabilistically — the model outputs a win probability, and a random draw determines the winner
3. Track which team wins each iteration
4. Divide each team's win count by 10,000 to get their championship probability

```
P(team wins Champions) = wins across 10,000 simulations / 10,000
```

This captures uncertainty across the full bracket — a team with 60% win probability in every match still loses sometimes, and those scenarios are all reflected in the final probabilities.

The same bracket logic also runs in two deterministic modes:

| Mode | How each match is decided | Used for |
|---|---|---|
| Monte Carlo | Random draw using the model probability | Title and playoff probability per team |
| Predicted bracket | The favorite always wins | The single most likely bracket, groups → Grand Final |
| Replay | Real result from vlr.gg | Rebuilding the real bracket to compare slot by slot |

---

## Champions 2026 Validation

`simulate_champions_2026.py` checks whether the approach actually works on Valorant Champions 2026 (Shanghai).

### How it avoids data leakage

The model is retrained with **only data from before Champions** (cutoff `2026-09-24`):

| Data | Source | Period |
|---|---|---|
| 2025 matches | Kaggle (piyush86kumar) | Kickoff → Champions 2025 |
| 2026 matches | vlr.gg via vlrdevapi | Kickoff → Stage 2 2026 (588 matches) |
| 2026 player stats | Kaggle (ryanluong1) | Kickoff → Masters London 2026 |
| Initial Elo | vlr.gg placements | Only events before 2025 (before the training period) |

Every feature (Elo, player stats) is frozen at the start of Champions. The model never sees any Champions result.

### What it outputs

1. **Ratings** — Elo of each team at Champions start
2. **Predicted bracket vs real** — the full bracket the model predicts (groups → Grand Final), next to the real one
3. **Model pick on real matchups** — accuracy on every Champions match already played
4. **Monte Carlo** — playoff and title probability per team

### Results (as of 2026-10-16, before the Lower Final)

| Metric | Model | Elo baseline | Before these changes (model) |
|---|---|---|---|
| Walk-forward log loss (lower is better) | 0.6757 | 0.6826 | 0.6667 (optimistic, leaked placements) |
| Accuracy on real matchups | 21/32 (65.6%) | 19/32 (59.4%) | 17/32 (53%) |
| Group stage accuracy | 16/20 | 15/20 | 14/20 |
| Playoffs accuracy | 5/12 | 4/12 | 3/12 |
| Playoff teams predicted correctly | 7/8 (missed LOUD; predicted EDG) | - | 7/8 |
| Predicted champion | Paper Rex (eliminated in Lower Round 1) | - | Paper Rex |

The old Elo baseline (0.8035 log loss) was not comparable: it used leaky initial Elo and no season regression. All 16 teams now have player stats at Champions.

The model's walk-forward log loss is the best of 50 Optuna trials selected on the same folds, so it is optimistic relative to the untuned Elo baseline.

### Known limitations

- **Elo still favors Paper Rex heavily.** Paper Rex starts Champions at 295 Elo, 65 points above the next team, while teams in form in 2026 (LOUD 8th, Vitality 10th) sit in the middle.
- **Low confidence.** Win probabilities on real matchups stay between 50% and 65%, so the model rarely separates teams clearly.
- **The edge over Elo is small.** Log loss improves by only 0.0069 (0.6757 vs 0.6826) and the model got two more matches right than the baseline (21 vs 19 of 32). With 32 matches the margin of error is about ±17 percentage points, so this is not evidence that the model beats Elo.

---

## Project Structure

```
champions-prediction-vlrdevapi/
│
├── src/
│   ├── config/
│   │   ├── constants.py          # paths, team IDs, weights, keywords
│   │   └── enums.py              # EventType, EventKeyword, BracketSlot
│   ├── ingestion/
│   │   ├── kaggle_loader.py      # downloads and organizes Kaggle datasets
│   │   ├── vlr_api.py            # fetches team placements via vlrdevapi
│   │   └── vlr_2026.py           # fetches 2026 VCT matches via vlrdevapi
│   ├── processing/
│   │   ├── normalizer.py         # normalizes matches into unified schema
│   │   ├── placements.py         # processes placements with time decay
│   │   └── player_stats_2026.py  # 2026 player stats from Kaggle
│   ├── features/
│   │   ├── elo.py                # initial and dynamic Elo calculation
│   │   ├── player_stats.py       # player stats aggregation (150-day window)
│   │   └── builder.py            # merges all features into final table
│   └── modeling/
│       ├── train.py              # walk-forward validation
│       ├── tuning.py             # Optuna hyperparameter optimization
│       ├── predict.py            # final model training and inference
│       ├── simulation.py         # bracket simulation: Monte Carlo, predicted, replay
│       ├── champions_2026.py     # retrains with pre-Champions data, win probability table
│       └── evaluation.py         # report tables (bracket comparison, accuracy, odds)
│
├── notebooks/
│   ├── EDA.ipynb                 # data exploration
│   └── api_exploration.ipynb     # vlrdevapi exploration and team ID mapping
│
├── main.py                       # runs the full pipeline
├── simulate_champions_2026.py    # Champions 2026 validation
├── data/                         # not tracked by git
│   ├── raw/
│   ├── processed/
│   └── features/
│
├── models/                       # not tracked by git
└── requirements.txt
```

---

## Data Sources

| Dataset | Author | Contents |
|---|---|---|
| [VCT 2025 - All Events](https://www.kaggle.com/datasets/piyush86kumar/valorant-vct-2025-all-events) | piyush86kumar | All 2025 events — Kickoff, Stage 1, Stage 2, Masters, Champions |
| [VCT 2021-2025](https://www.kaggle.com/datasets/ryanluong1/valorant-champion-tour-2021-2023-data) | ryanluong1 | Historical match and player data 2021–2026 (2026 up to Masters London) |
| [vlr.gg](https://www.vlr.gg) via [vlrdevapi](https://pypi.org/project/vlrdevapi/) | — | Team placements and 2026 matches with dates |

> Data is not included in this repository. See **How to Run** below.

---

## How to Run

**1. Clone the repository**
```bash
git clone https://github.com/DaviAlcanfor/champions-prediction-vlrdevapi.git
cd champions-prediction-vlrdevapi
```

**2. Create and activate virtual environment**
```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Linux/Mac
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Configure Kaggle API**

Download `kaggle.json` from [kaggle.com/settings](https://www.kaggle.com/settings) → API → Create New Token, and place it at:
- Windows: `C:\Users\USERNAME\.kaggle\kaggle.json`
- Linux/Mac: `~/.kaggle/kaggle.json`

**5. Run the pipeline**
```bash
python main.py
```

**6. Run the Champions 2026 validation**
```bash
python simulate_champions_2026.py
```
Fetches the latest results from vlr.gg on every run, so it can be re-run as Champions goes on.
