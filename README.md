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

Each team's starting Elo is calculated from their historical placements across all VCT events, weighted by two factors:

| Factor | Logic |
|---|---|
| Event importance | Champions > Masters > Regional |
| Time decay | Recent results matter more (`half_life = 2 years`) |

```
weighted_score = event_weight × decay / place
initial_elo   = sum(weighted_scores) × 100
```

---

### Player Stats

| Feature | Description |
|---|---|
| ACS / KAST / ADR | Aggregated per team over a 60-day window before each match |
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

*(Waiting for Valorant Champions 2026 participant list)*

Once the bracket is confirmed, the simulation works as follows:

1. For each of 10,000 iterations, simulate the full tournament bracket from start to finish
2. Each matchup is decided probabilistically — the model outputs a win probability, and a random draw determines the winner
3. Track which team wins each iteration
4. Divide each team's win count by 10,000 to get their championship probability

```
P(team wins Champions) = wins across 10,000 simulations / 10,000
```

This captures uncertainty across the full bracket — a team with 60% win probability in every match still loses sometimes, and those scenarios are all reflected in the final probabilities.

---

## Project Structure

```
champions-prediction-vlrdevapi/
│
├── src/
│   ├── config/
│   │   ├── constants.py          # paths, team IDs, weights, keywords
│   │   └── enums.py              # EventType, EventKeyword
│   ├── ingestion/
│   │   ├── kaggle_loader.py      # downloads and organizes Kaggle datasets
│   │   └── vlr_api.py            # fetches team placements via vlrdevapi
│   ├── processing/
│   │   ├── normalizer.py         # normalizes matches into unified schema
│   │   └── placements.py         # processes placements with time decay
│   ├── features/
│   │   ├── elo.py                # initial and dynamic Elo calculation
│   │   ├── player_stats.py       # player stats aggregation (60-day window)
│   │   └── builder.py            # merges all features into final table
│   └── modeling/
│       ├── train.py              # walk-forward validation
│       ├── tuning.py             # Optuna hyperparameter optimization
│       └── predict.py            # final model training and inference
│
├── notebooks/
│   ├── EDA.ipynb                 # data exploration
│   ├── api_exploration.ipynb     # vlrdevapi exploration and team ID mapping
│   └── simulation.ipynb          # Monte Carlo simulation and results *(coming soon)*
│
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
| [VCT 2021-2025](https://www.kaggle.com/datasets/ryanluong1/valorant-champion-tour-2021-2023-data) | ryanluong1 | Historical match and player data 2021–2025 |

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
