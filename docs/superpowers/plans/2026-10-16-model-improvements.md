# Model Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fazer o modelo ser medido contra um baseline honesto e corrigir o que impede ele de superar o Elo puro: vazamento no Elo inicial, Elo preso na temporada anterior, feature redundante e stats de jogador que não chegam à Champions.

**Architecture:** Mudanças pequenas nos módulos existentes (`elo.py`, `placements.py`, `train.py`, `tuning.py`, `player_stats.py`, `champions_2026.py`, `evaluation.py`), sem módulos novos. Testes mínimos em `tests/`, um arquivo por módulo, só com o que quebra se a lógica quebrar.

**Tech Stack:** Python 3.13, pandas, XGBoost, Optuna, pytest (já no `requirements.txt`).

**Spec:** sem documento separado; requisitos combinados no chat em 2026-10-16 (avaliação do projeto + respostas do usuário).

## Global Constraints

- Seguir o padrão do projeto: dados fixos em `src/config/constants.py` / `enums.py`; funções com docstring curta; sem classes; blocos com linha em branco entre si.
- Nada de código "por via das dúvidas": sem tratamento de erro para cenário impossível, sem configuração que ninguém pediu.
- Testes: pytest puro, sem fixtures nem frameworks extras. Rodar sempre com `.venv/Scripts/python -m pytest tests -q` a partir da raiz.
- Janela de stats de jogador: **150 dias** (com 120 só 8 dos 16 times da Champions têm stats; com 150, os 16).
- Regressão do Elo entre temporadas: **25%** da distância até a média.
- Seed fixa: **42**.
- Trabalhar na branch `feat/model-improvements`, um commit por task, mensagem terminando com `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## File Structure

| Arquivo | O que muda |
|---|---|
| `src/modeling/train.py` | `walk_forward_folds()` (tira a duplicação do loop de folds), `elo_baseline_metrics()`, `elo_baseline_log_loss()`; remove `win_prob_team1` de `FEATURE_COLS` |
| `src/modeling/tuning.py` | usa `walk_forward_folds()`; imprime o baseline; seed no Optuna |
| `src/processing/placements.py` | `placements_before()` |
| `src/features/elo.py` | `regress_to_mean()`; regressão entre temporadas em `calculate_dynamic_elo()`; `__main__` usa `placements_before()` |
| `src/features/player_stats.py` | `WINDOW_DAYS = 150` + docstrings |
| `src/modeling/champions_2026.py` | Elo inicial sem vazamento; baseline no retorno; remove feature redundante; seed |
| `src/modeling/evaluation.py` | coluna `elo_hit` em `match_predictions()` |
| `src/modeling/simulation.py` | seed default vem de `RANDOM_SEED` |
| `src/config/constants.py` | `SEASON_REGRESSION`, `FEATURES_PATH`, `MODELS_PATH`, `RANDOM_SEED` |
| `main.py` | cria as pastas de saída |
| `simulate_champions_2026.py` | imprime baseline ao lado do modelo |
| `requirements.txt` | UTF-16 → UTF-8 |
| `README.md` | janela, baseline, pastas, resultados novos |
| `tests/test_train.py`, `tests/test_placements.py`, `tests/test_elo.py`, `tests/test_player_stats.py`, `tests/test_simulation.py` | novos |

---

### Task 0: Branch

- [ ] **Step 1: Criar a branch a partir da main atualizada**

```bash
git checkout main && git pull -q origin main && git checkout -b feat/model-improvements
```

---

### Task 1: Baseline de Elo

O modelo só serve se superar a fórmula de Elo sozinha. Hoje ninguém mede isso. O loop de folds está copiado em `train.py` e `tuning.py`; o baseline seria a terceira cópia, então ele vira uma função.

**Files:**
- Modify: `src/modeling/train.py` (novas funções após `fill_missing_stats`; `walk_forward_validation` e `__main__`)
- Modify: `src/modeling/tuning.py` (loop do `objective`, `__main__`)
- Modify: `src/modeling/champions_2026.py` (`train_champions_model`)
- Modify: `src/modeling/evaluation.py` (`match_predictions`)
- Modify: `simulate_champions_2026.py`
- Test: `tests/test_train.py`

**Interfaces:**
- Produces: `walk_forward_folds(df: pd.DataFrame) -> Iterator[tuple[str, pd.DataFrame, pd.DataFrame]]` (test_event, train, test); `elo_baseline_metrics(df: pd.DataFrame) -> dict` com chaves `elo_accuracy`, `elo_log_loss`; `elo_baseline_log_loss(df: pd.DataFrame) -> float`; `train_champions_model()` passa a devolver também a chave `"elo_log_loss"`; `match_predictions(champions, win_prob, elo)` ganha a coluna `elo_hit`.

- [ ] **Step 1: Escrever os testes que falham**

`tests/test_train.py`:

```python
import pandas as pd
import pytest
from src.modeling.train import elo_baseline_metrics, walk_forward_folds


def test_elo_baseline_metrics_scores_the_elo_favorite():
    df = pd.DataFrame({
        "win_prob_team1": [0.8, 0.3, 0.6, 0.4],
        "result": [1, 0, 0, 1],
    })

    metrics = elo_baseline_metrics(df)

    assert metrics["elo_accuracy"] == 0.5
    assert metrics["elo_log_loss"] == pytest.approx(0.6031, abs=1e-4)


def test_walk_forward_folds_never_train_on_future_events():
    df = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=30, freq="D"),
        "event": ["A"] * 12 + ["B"] * 12 + ["C"] * 6,
    })

    folds = list(walk_forward_folds(df))

    assert [event for event, _, _ in folds] == ["B", "C"]
    for _, train, test in folds:
        assert train["date"].max() < test["date"].min()
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `.venv/Scripts/python -m pytest tests/test_train.py -q`
Expected: FAIL com `ImportError: cannot import name 'elo_baseline_metrics'`

- [ ] **Step 3: Implementar em `src/modeling/train.py`**

Logo após `fill_missing_stats`:

```python
def walk_forward_folds(df: pd.DataFrame):
    """
    Yields (test_event, train, test): trains on all past tournaments, tests on the next one.
    Skips folds too small to evaluate.
    """
    events = get_tournament_order(df)

    for i in range(1, len(events)):
        train = df[df["event"].isin(events[:i])]
        test = df[df["event"] == events[i]]

        if len(train) < 10 or len(test) < 3:
            continue

        yield events[i], train, test


def elo_baseline_metrics(df: pd.DataFrame) -> dict:
    """
    Accuracy and log loss of the Elo formula alone (win_prob_team1), no model.
    The model is only worth it if it beats these numbers.
    """
    y_true = df[TARGET_COL]
    y_prob = df["win_prob_team1"]

    return {
        "elo_accuracy": round(accuracy_score(y_true, (y_prob >= 0.5).astype(int)), 4),
        "elo_log_loss": round(log_loss(y_true, y_prob, labels=[0, 1]), 4),
    }


def elo_baseline_log_loss(df: pd.DataFrame) -> float:
    """Mean walk-forward log loss of the Elo formula alone, comparable to the tuning objective."""
    return float(np.mean([
        elo_baseline_metrics(test)["elo_log_loss"]
        for _, _, test in walk_forward_folds(df)
    ]))
```

Em `walk_forward_validation`, trocar o começo do loop (de `events = get_tournament_order(df)` até o `continue`) por:

```python
    results = []

    for test_event, train, test in walk_forward_folds(df):
        X_train = train[FEATURE_COLS]
```

(as linhas de `X_train` em diante continuam iguais) e no `results.append` adicionar a última chave:

```python
            "log_loss": round(log_loss(y_test, y_prob), 4),
            **elo_baseline_metrics(test),
        })
```

No `__main__`, depois de `Mean log_loss`:

```python
    print(f"Elo baseline accuracy: {results_df['elo_accuracy'].mean():.4f}")
    print(f"Elo baseline log_loss: {results_df['elo_log_loss'].mean():.4f}")
```

- [ ] **Step 4: Usar `walk_forward_folds` em `src/modeling/tuning.py`**

Import: trocar `get_tournament_order` por `walk_forward_folds, elo_baseline_log_loss`.

No `objective`, trocar o bloco de `events = get_tournament_order(df)` até o `continue` por:

```python
    fold_losses = []

    for _, train, test in walk_forward_folds(df):
        X_train = train[FEATURE_COLS]
```

No `__main__`, depois de `Best log_loss`:

```python
    print(f"Elo baseline log_loss: {elo_baseline_log_loss(df):.4f}")
```

- [ ] **Step 5: Baseline na validação da Champions**

`src/modeling/champions_2026.py`: importar `elo_baseline_log_loss` de `src.modeling.train` e, no dict que `train_champions_model` devolve, adicionar:

```python
        "elo_log_loss": elo_baseline_log_loss(features),
```

`src/modeling/evaluation.py`: `match_predictions` recebe `elo` e marca se o favorito pelo Elo venceu:

```python
def match_predictions(champions: pd.DataFrame, win_prob: WinProbTable, elo: dict) -> pd.DataFrame:
    """Model pick and Elo favorite for every real Champions matchup already played."""
    rows = []

    for _, match in champions.iterrows():
        team1, team2 = match["team1"], match["team2"]
        p_team1 = win_prob[(team1, team2)]

        pick, prob = (team1, p_team1) if p_team1 >= 0.5 else (team2, 1 - p_team1)
        elo_favorite = team1 if elo.get(team1, BASE_ELO) >= elo.get(team2, BASE_ELO) else team2

        rows.append({
            "stage": match["stage"],
            "match": f"{team1} x {team2}",
            "pick": pick,
            "prob_%": _to_percent(prob),
            "winner": match["winner"],
            "hit": pick == match["winner"],
            "elo_hit": elo_favorite == match["winner"],
        })

    return pd.DataFrame(rows)
```

`simulate_champions_2026.py`: depois de `Best walk-forward log_loss`:

```python
    print(f"Elo baseline log_loss:     {trained['elo_log_loss']:.4f}")
```

e trocar o bloco de predictions por:

```python
    predictions = match_predictions(champions, win_prob, trained["elo"])
    print_section("Model pick on real matchups", predictions)
    print_accuracy("Model accuracy", predictions["hit"])
    print_accuracy("Elo baseline accuracy", predictions["elo_hit"])
```

- [ ] **Step 6: Rodar os testes e o tuning**

Run: `.venv/Scripts/python -m pytest tests -q`
Expected: 2 passed

Run: `.venv/Scripts/python -W ignore -m src.modeling.tuning 2>&1 | tail -4`
Expected: imprime `Best log_loss` e `Elo baseline log_loss` (anotar os dois: é o "antes")

- [ ] **Step 7: Commit**

```bash
git add src/modeling/train.py src/modeling/tuning.py src/modeling/champions_2026.py src/modeling/evaluation.py simulate_champions_2026.py tests/test_train.py
git commit -m "feat: Elo-only baseline in walk-forward and Champions report

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Elo inicial sem vazamento

Hoje o Elo inicial usa as colocações de 2025, e o modelo é treinado e validado em 2025. Ele começa a temporada sabendo quem vai bem nela. O corte passa a ser o ano da primeira partida de treino; isso também elimina automaticamente a Champions 2026.

**Files:**
- Modify: `src/processing/placements.py` (nova função após `process_placements`)
- Modify: `src/features/elo.py` (import, `__main__`)
- Modify: `src/modeling/champions_2026.py` (`_initial_elo_without_champions` → `_initial_elo`)
- Test: `tests/test_placements.py`

**Interfaces:**
- Produces: `placements_before(placements_df: pd.DataFrame, year: int) -> pd.DataFrame`

- [ ] **Step 1: Escrever o teste que falha**

`tests/test_placements.py`:

```python
import pandas as pd
from src.processing.placements import placements_before


def test_placements_before_drops_cutoff_year_and_later():
    df = pd.DataFrame({
        "team": ["LOUD", "LOUD", "LOUD"],
        "year": [2023, 2024, 2025],
        "weighted_score": [1.0, 1.0, 1.0],
    })

    assert placements_before(df, 2025)["year"].tolist() == [2023, 2024]
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `.venv/Scripts/python -m pytest tests/test_placements.py -q`
Expected: FAIL com `ImportError: cannot import name 'placements_before'`

- [ ] **Step 3: Implementar em `src/processing/placements.py`**

Depois de `process_placements`:

```python
def placements_before(placements_df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Keeps only placements from before `year`.
    Initial Elo must not know results of the season it is used to predict.
    """
    return placements_df[placements_df["year"] < year]
```

- [ ] **Step 4: Usar no `__main__` de `src/features/elo.py`**

Import: `from src.processing.placements import classify_event, placements_before`.

Trocar `initial_elo = calculate_initial_elo(placements_df)` por:

```python
    first_year = matches_df["date"].min().year
    initial_elo = calculate_initial_elo(placements_before(placements_df, first_year))
```

- [ ] **Step 5: Usar em `src/modeling/champions_2026.py`**

Import: `from src.processing.placements import placements_before`.

Substituir a função `_initial_elo_without_champions` inteira por:

```python
def _initial_elo(matches: pd.DataFrame) -> dict[str, float]:
    """Initial Elo from placements before the first training match (no leakage)."""
    placements = pd.read_parquet(os.path.join(PROCESSED_PATH, "placements.parquet"))
    first_year = matches["date"].min().year

    return calculate_initial_elo(placements_before(placements, first_year))
```

e em `build_training_data`:

```python
    elo_df = calculate_dynamic_elo(matches, _initial_elo(matches))
```

- [ ] **Step 6: Rodar testes**

Run: `.venv/Scripts/python -m pytest tests -q`
Expected: 3 passed

- [ ] **Step 7: Commit**

```bash
git add src/processing/placements.py src/features/elo.py src/modeling/champions_2026.py tests/test_placements.py
git commit -m "fix: initial Elo only uses placements before the training period

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Regressão do Elo entre temporadas

Elencos mudam todo ano, mas o Elo carrega a temporada anterior inteira (a Paper Rex entrou na Champions 2026 com 120 pontos de vantagem por causa de 2025). Na virada do ano, cada time perde 25% da distância até a média.

**Files:**
- Modify: `src/config/constants.py` (depois de `STAGE_K`)
- Modify: `src/features/elo.py` (nova função antes de `calculate_dynamic_elo`; loop de `calculate_dynamic_elo`)
- Test: `tests/test_elo.py`

**Interfaces:**
- Consumes: `SEASON_REGRESSION` de `src.config.constants`
- Produces: `regress_to_mean(elo: dict[str, float], factor: float) -> dict[str, float]`

- [ ] **Step 1: Escrever os testes que falham**

`tests/test_elo.py`:

```python
import pandas as pd
import pytest
from src.config.constants import SEASON_REGRESSION
from src.features.elo import calculate_dynamic_elo, regress_to_mean


def test_regress_to_mean_pulls_ratings_toward_average():
    assert regress_to_mean({"A": 100.0, "B": 0.0}, 0.25) == {"A": 87.5, "B": 12.5}


def test_dynamic_elo_regresses_between_seasons():
    matches = pd.DataFrame({
        "date": pd.to_datetime(["2025-06-01", "2026-01-10"]),
        "team1": ["A", "A"],
        "team2": ["B", "B"],
        "winner": ["A", "A"],
        "stage": ["Playoffs", "Playoffs"],
        "event": ["VCT 2025: EMEA Stage 1", "VCT 2026: EMEA Kickoff"],
    })

    elo_df = calculate_dynamic_elo(matches, {"A": 100.0, "B": 0.0})
    first, second = elo_df.iloc[0], elo_df.iloc[1]

    mean = (first["elo1_after"] + first["elo2_after"]) / 2
    expected_a = mean + (first["elo1_after"] - mean) * (1 - SEASON_REGRESSION)

    assert second["elo1_before"] == pytest.approx(expected_a, abs=0.01)
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `.venv/Scripts/python -m pytest tests/test_elo.py -q`
Expected: FAIL com `ImportError: cannot import name 'SEASON_REGRESSION'`

- [ ] **Step 3: Constante em `src/config/constants.py`**

Depois do dict `STAGE_K`:

```python
# Share of the distance to the mean Elo every team loses when a new season starts
SEASON_REGRESSION = 0.25
```

- [ ] **Step 4: Implementar em `src/features/elo.py`**

Import: `from src.config.constants import STAGE_K, EVENT_WEIGHTS, SEASON_REGRESSION`.

Antes de `calculate_dynamic_elo`:

```python
def regress_to_mean(elo: dict[str, float], factor: float) -> dict[str, float]:
    """
    Pulls every team's Elo toward the mean by `factor` (0 = unchanged, 1 = everyone equal).
    Applied between seasons: rosters change, so last season's rating is only partly valid.
    """
    mean = sum(elo.values()) / len(elo)

    return {team: mean + (rating - mean) * (1 - factor) for team, rating in elo.items()}
```

Em `calculate_dynamic_elo`, trocar o início:

```python
    elo = dict(initial_elo)
    season = None
    rows = []

    for _, match in matches_df.iterrows():
        # new season: ratings partly reset toward the mean
        if season is not None and match["date"].year != season:
            elo = regress_to_mean(elo, SEASON_REGRESSION)
        season = match["date"].year

        team1 = match["team1"]
```

(o resto do loop continua igual)

- [ ] **Step 5: Rodar testes**

Run: `.venv/Scripts/python -m pytest tests -q`
Expected: 5 passed

- [ ] **Step 6: Commit**

```bash
git add src/config/constants.py src/features/elo.py tests/test_elo.py
git commit -m "feat: regress Elo toward the mean between seasons

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Remover a feature redundante

`win_prob_team1` é uma função monotônica de `elo_diff`: para uma árvore, as duas colunas carregam a mesma informação. A coluna continua sendo calculada em `elo.py`, porque o baseline (Task 1) usa ela.

**Files:**
- Modify: `src/modeling/train.py` (`FEATURE_COLS`)
- Modify: `src/modeling/champions_2026.py` (`_matchup_features`, import de `expected_score`)

- [ ] **Step 1: Tirar de `FEATURE_COLS` em `src/modeling/train.py`**

```python
FEATURE_COLS = [
    "elo_diff",
    "acs_team1", "kast_team1", "adr_team1",
    "acs_team2", "kast_team2", "adr_team2",
]
```

- [ ] **Step 2: Tirar de `_matchup_features` em `src/modeling/champions_2026.py`**

```python
    row = {"elo_diff": elo_a - elo_b}
```

e remover `expected_score` do import de `src.features.elo` (fica sem uso).

- [ ] **Step 3: Rodar testes e o tuning**

Run: `.venv/Scripts/python -m pytest tests -q`
Expected: 5 passed

Run: `.venv/Scripts/python -W ignore -m src.modeling.tuning 2>&1 | tail -4`
Expected: roda sem erro; `Best log_loss` na mesma faixa de antes (a remoção não deve piorar)

- [ ] **Step 4: Commit**

```bash
git add src/modeling/train.py src/modeling/champions_2026.py
git commit -m "refactor: drop win_prob_team1 feature, redundant with elo_diff

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Janela de stats de 150 dias

Com 60 dias, nenhum time chega à Champions com stats (o Stage 2 não tem stats no Kaggle). Medido em 2026-10-16: 120 dias cobre 8 dos 16 times; 150 cobre os 16 (os Stage 1 regionais terminaram entre 10/05 e 24/05).

**Files:**
- Modify: `src/features/player_stats.py:6` e docstrings das linhas 46 e 74
- Modify: `src/modeling/champions_2026.py` (docstring de `team_stats_at_start`)
- Test: `tests/test_player_stats.py`

- [ ] **Step 1: Escrever o teste**

`tests/test_player_stats.py`:

```python
import pandas as pd
from src.features.player_stats import WINDOW_DAYS, compute_team_stats


def test_compute_team_stats_uses_only_the_window_before_the_match():
    match_date = pd.Timestamp("2026-09-24")
    player_df = pd.DataFrame({
        "match_date": [
            match_date - pd.Timedelta(days=WINDOW_DAYS - 1),  # inside the window
            match_date - pd.Timedelta(days=WINDOW_DAYS + 1),  # too old
            match_date,                                       # the match itself: future
        ],
        "player_team": ["LOUD"] * 3,
        "acs": [200.0, 999.0, 999.0],
        "kast": [70.0, 0.0, 0.0],
        "adr": [130.0, 0.0, 0.0],
    })

    assert compute_team_stats(player_df, match_date, "LOUD") == {"acs": 200.0, "kast": 70.0, "adr": 130.0}


def test_window_reaches_stage_1_before_champions_2026():
    # Stage 1 2026 ended on 2026-05-10 (China); Champions started on 2026-09-24
    assert pd.Timestamp("2026-09-24") - pd.Timedelta(days=WINDOW_DAYS) <= pd.Timestamp("2026-05-10")
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `.venv/Scripts/python -m pytest tests/test_player_stats.py -q`
Expected: o primeiro passa, `test_window_reaches_stage_1_before_champions_2026` FALHA (janela ainda é 60)

- [ ] **Step 3: Mudar a janela em `src/features/player_stats.py`**

```python
WINDOW_DAYS = 150
```

e nas duas docstrings trocar "the 60 days before" por "the WINDOW_DAYS before":

```python
    Computes mean ACS, KAST, ADR for a team in the WINDOW_DAYS before a given match date.
```

```python
    Uses only data from the WINDOW_DAYS before each match to avoid leakage.
```

Em `src/modeling/champions_2026.py`, docstring de `team_stats_at_start`:

```python
    """Mean ACS/KAST/ADR of each team in the WINDOW_DAYS before Champions ({} if no data)."""
```

- [ ] **Step 4: Rodar testes**

Run: `.venv/Scripts/python -m pytest tests -q`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/features/player_stats.py src/modeling/champions_2026.py tests/test_player_stats.py
git commit -m "feat: widen player stats window to 150 days

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Testes da simulação

A lógica da chave é pura e fácil de quebrar sem perceber (ordem do cruzamento na chave inferior, 1º e 2º do grupo). Só testes, sem mudar código.

**Files:**
- Test: `tests/test_simulation.py`

- [ ] **Step 1: Escrever os testes**

`tests/test_simulation.py`:

```python
from src.config.constants import CHAMPIONS_2026_GROUPS
from src.modeling.simulation import TBD, predicted_bracket, replay_bracket, simulate_group

TEAMS = [team for group in CHAMPIONS_2026_GROUPS.values() for team in group]


def test_group_first_is_winners_match_winner_and_second_is_decider_winner():
    # alphabetical order decides every match: A beats everyone, B beats C and D...
    def play(slot, team_a, team_b):
        return min(team_a, team_b), max(team_a, team_b)

    assert simulate_group(play, "X", ("A", "B", "C", "D")) == ("A", "B")


def test_predicted_bracket_crowns_the_team_that_beats_everyone():
    win_prob = {}
    for a in TEAMS:
        for b in TEAMS:
            if a != b:
                win_prob[(a, b)] = 0.9 if a == "LOUD" else 0.1 if b == "LOUD" else 0.5

    bracket = predicted_bracket(win_prob, CHAMPIONS_2026_GROUPS)

    assert len(bracket) == 34  # 4 groups x 5 matches + 14 playoff matches
    assert bracket[-1]["winner"] == "LOUD"


def test_replay_without_results_leaves_every_match_tbd():
    bracket = replay_bracket([], CHAMPIONS_2026_GROUPS)

    assert all(match["winner"] == TBD for match in bracket)
```

- [ ] **Step 2: Rodar**

Run: `.venv/Scripts/python -m pytest tests -q`
Expected: 10 passed (os de simulação passam de primeira: cobrem código que já funciona; se algum falhar, é bug real na chave)

- [ ] **Step 3: Commit**

```bash
git add tests/test_simulation.py
git commit -m "test: bracket simulation (groups, predicted bracket, replay)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Correções de engenharia

Clone limpo quebra sem as pastas; Optuna sem seed dá resultado diferente a cada rodada; `requirements.txt` em UTF-16 aparece como binário no GitHub.

**Files:**
- Modify: `src/config/constants.py` (bloco "Folder paths")
- Modify: `main.py`
- Modify: `src/modeling/tuning.py:72`, `src/modeling/champions_2026.py` (`tune_and_train`), `src/modeling/simulation.py` (`monte_carlo`)
- Modify: `requirements.txt`, `README.md`

**Interfaces:**
- Produces: `FEATURES_PATH`, `MODELS_PATH`, `RANDOM_SEED` em `src.config.constants`

- [ ] **Step 1: Constantes em `src/config/constants.py`**

No bloco `# Folder paths`, logo após `PROCESSED_PATH`:

```python
FEATURES_PATH = os.path.join("data", "features")
MODELS_PATH = "models"

RANDOM_SEED = 42
```

- [ ] **Step 2: `main.py` cria as pastas**

```python
import os
import subprocess
import sys
from src.config.constants import PROCESSED_PATH, FEATURES_PATH, MODELS_PATH

OUTPUT_DIRS = [PROCESSED_PATH, FEATURES_PATH, MODELS_PATH]

PIPELINE = [
    ...  # lista atual, sem mudança
]

if __name__ == "__main__":

    for path in OUTPUT_DIRS:
        os.makedirs(path, exist_ok=True)

    for module in PIPELINE:
        ...  # loop atual, sem mudança
```

- [ ] **Step 3: Seed no Optuna e no Monte Carlo**

`src/modeling/tuning.py` e `src/modeling/champions_2026.py` (importar `RANDOM_SEED` de `src.config.constants`):

```python
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
```

`src/modeling/simulation.py` (importar `RANDOM_SEED`): na assinatura de `monte_carlo`, `seed: int = RANDOM_SEED`.

- [ ] **Step 4: `requirements.txt` para UTF-8**

```bash
iconv -f UTF-16 -t UTF-8 requirements.txt | sed '1s/^\xEF\xBB\xBF//' > requirements.tmp && mv requirements.tmp requirements.txt
file requirements.txt
.venv/Scripts/python -m pip install --dry-run -q -r requirements.txt
```

Expected: `file` mostra `ASCII text` (ou UTF-8); o dry-run termina sem erro.

- [ ] **Step 5: README**

Em "How to Run", passo 5, remover a linha `mkdir data\features models   # not created by the pipeline yet`.

- [ ] **Step 6: Verificar reprodutibilidade**

```bash
rm -rf data/features models
.venv/Scripts/python main.py > /dev/null 2>&1 && ls data/features models
.venv/Scripts/python -W ignore simulate_champions_2026.py 2>&1 | grep -v fetched > run1.txt
.venv/Scripts/python -W ignore simulate_champions_2026.py 2>&1 | grep -v fetched > run2.txt
diff run1.txt run2.txt && echo IDENTICAL
.venv/Scripts/python -m pytest tests -q
```

Expected: as pastas aparecem; `IDENTICAL`; 10 passed. Depois apagar `run1.txt run2.txt`.

- [ ] **Step 7: Commit**

```bash
git add src/config/constants.py main.py src/modeling/tuning.py src/modeling/champions_2026.py src/modeling/simulation.py requirements.txt README.md
git commit -m "chore: pipeline creates output dirs, fixed seeds, requirements in UTF-8

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Rodar tudo, comparar e documentar

**Files:**
- Modify: `README.md` (Player Stats, Elo, Champions 2026 Validation)

- [ ] **Step 1: Rodar o pipeline e a validação**

```bash
.venv/Scripts/python main.py 2>&1 | grep -E "Best log_loss|Elo baseline"
.venv/Scripts/python -W ignore simulate_champions_2026.py 2>&1 | grep -v fetched > results.txt
```

Anotar de `results.txt`: walk-forward log loss do modelo e do baseline; acertos do modelo e do baseline na Champions (total, grupos, playoffs); classificados acertados; campeão previsto; quantos times têm stats (`has_player_stats`).

- [ ] **Step 2: Atualizar o README com os números do Step 1**

- Tabela de Player Stats: `Aggregated per team over a 150-day window before each match`.
- Seção Elo: um parágrafo "Season regression" explicando os 25% (`SEASON_REGRESSION`) e um "Initial Elo" dizendo que só usa colocações de anos anteriores ao treino.
- "Champions 2026 Validation" → tabela "Results" com uma coluna para o modelo e outra para o baseline de Elo, preenchidas com os números do Step 1.
- "Known limitations": remover os itens resolvidos (stats sem cobertura, Elo preso em 2025) e manter o que continuar verdade segundo os números novos. Se o modelo não superar o baseline, dizer isso explicitamente.

- [ ] **Step 3: Testes finais e limpeza**

```bash
.venv/Scripts/python -m pytest tests -q
rm results.txt
git status --short
```

Expected: 10 passed; só `README.md` modificado.

- [ ] **Step 4: Commit e PR**

```bash
git add README.md
git commit -m "docs: results after model improvements, model vs Elo baseline

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push -u origin feat/model-improvements
gh pr create --base main --title "Model improvements: Elo baseline, leakage-free initial Elo, season regression"
```

O corpo do PR deve trazer a tabela modelo vs baseline (antes e depois) e terminar com `🤖 Generated with [Claude Code](https://claude.com/claude-code)`. O merge fica com o usuário.
