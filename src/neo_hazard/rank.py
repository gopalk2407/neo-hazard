"""Score every object and write the ranked hazard list."""
import joblib
import numpy as np
import pandas as pd

from .data import ROOT, load
from .features import SIGMA_COLS, build_features, impact_proxy
from .train import MODEL_PATH

OUT_PATH = ROOT / "outputs" / "hazard_ranking.csv"
SCORED_PATH = ROOT / "outputs" / "scored.parquet"


def rank(top_n: int = 50) -> pd.DataFrame:
    df = load()
    model = joblib.load(MODEL_PATH)
    df["p_pha_ml"] = model.predict_proba(build_features(df))[:, 1]
    df = df.join(impact_proxy(df))

    # Combined score: ML hazard probability x physical encounter proxy.
    # Rows without a catalogue MOID fall back to the ML probability alone.
    df["hazard_score"] = df["p_pha_ml"] * df["impact_proxy_per_yr"].fillna(0)
    df = df.sort_values(["hazard_score", "p_pha_ml"], ascending=False)

    cols = ["full_name", "class", "neo", "pha", "H", "diameter", "moid", "moid_ld",
            "moid_sigma_au", "capture_radius_au", "p_moid_within_capture",
            "impact_proxy_per_yr", "p_pha_ml", "hazard_score"]
    df[cols].to_csv(OUT_PATH, index=False)
    keep = cols + ["pdes", "e", "a", "q", "i", "om", "w", "rms", *SIGMA_COLS]
    df[keep].reset_index(drop=True).to_parquet(SCORED_PATH, index=False)

    unlabeled = df[df["pha"].isna()].sort_values("p_pha_ml", ascending=False)
    unlabeled[["full_name", "class", "H", "e", "a", "q", "i", "p_pha_ml"]].head(top_n) \
        .to_csv(OUT_PATH.with_name("unlabeled_candidates.csv"), index=False)
    return df[cols].head(top_n)


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    print(rank(25).to_string(index=False))
