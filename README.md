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
