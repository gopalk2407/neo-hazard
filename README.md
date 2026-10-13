# neo-hazard

Rank objects in the JPL Small-Body Database export (`dataset.csv`, ~958k rows) by
how likely they are to collide with Earth.

```
uv sync
uv run python -m neo_hazard.train   # trains model -> models/, metrics -> outputs/metrics.json
uv run python -m neo_hazard.rank    # writes outputs/hazard_ranking.csv, outputs/unlabeled_candidates.csv
uv run streamlit run app.py         # local web UI at http://localhost:8501
```

The web UI has three modes: pick a known NEO, enter orbital elements (MOID is computed
in `moid.py`), or upload a CSV with columns `a,e,i,om,w,H` (optional `name`, `moid`,
`sigma_*`). It shows the hazard probability, collision metrics, an orbit plot, and a
MOID-vs-H chart against all known NEOs.

## Running on another Mac

The only software to install by hand is **uv**; it fetches the right Python (3.12) and
every dependency itself. No Homebrew system libraries are required.

```bash
# 1. Install uv (pick one)
curl -LsSf https://astral.sh/uv/install.sh | sh    # official installer
brew install uv                                     # or via Homebrew

# 2. Get the code, then install dependencies into a local .venv
git clone <repo-url> neo-hazard && cd neo-hazard    # or copy the folder
uv sync
```

`dataset.csv`, `models/` and `outputs/` are git-ignored (too large), so a fresh clone
has only the code. Choose one of:

**A. Just run the app** (no 435 MB dataset needed). Copy two generated files from a
machine that already has them — `models/pha_hgb.joblib` (5 MB) and
`outputs/scored.parquet` (160 MB) — into the matching folders, then:

```bash
uv run streamlit run app.py          # http://localhost:8501
```

**B. Regenerate everything from scratch.** Copy `dataset.csv` (435 MB) into the project
root, then build the model and scores (about 1–2 minutes total) and launch:

```bash
uv run python -m neo_hazard.train    # -> models/, outputs/metrics.json
uv run python -m neo_hazard.rank     # -> outputs/scored.parquet, rankings
uv run streamlit run app.py
```

## Approach
1. **ML hazard classifier** (`train.py`): gradient-boosted trees predict P(PHA) from
   orbital elements, H, and orbit uncertainties. MOID is *excluded*: PHA is defined as
   `MOID <= 0.05 AU and H <= 22`, so including it would just leak the label rule.
   Engineered features (`features.py`) include node distances from 1 AU, Tisserand
   parameter, and Öpik encounter speed.
2. **Physical impact proxy** (`features.impact_proxy`): probability that the true MOID
   lies within Earth's gravitationally focused capture radius (given MOID uncertainty),
   times orbits per year.
3. **hazard_score** = P(PHA) × impact proxy, for ranking.

This is a relative ranking heuristic, not a true impact probability; those require
orbit propagation (see NASA CNEOS Sentry).
