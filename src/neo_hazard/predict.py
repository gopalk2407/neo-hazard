"""Score user-supplied orbits with the trained model + impact proxy."""
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from .features import SIGMA_COLS, build_features, impact_proxy
from .moid import moid as compute_moid
from .train import MODEL_PATH

GAUSS_K_DEG = 0.9856076686  # mean motion (deg/day) of a 1 AU orbit
REQUIRED = ["a", "e", "i", "om", "w", "H"]
# Typical error of moid.moid() vs JPL MOIDs (median 1e-4, 95th pct 6e-4 AU on NEOs);
# it uses mean J2000 Earth elements rather than Earth's orbit at each object's epoch.
COMPUTED_MOID_SIGMA_AU = 3e-4


@lru_cache(maxsize=1)
def model():
    return joblib.load(MODEL_PATH)


def complete_elements(df: pd.DataFrame, defaults: dict) -> pd.DataFrame:
    """Derive q, ad, n, per_y, MOID; fill missing uncertainties with defaults."""
    df = df.copy()
    df["q"] = df["a"] * (1 - df["e"])
    df["ad"] = df["a"] * (1 + df["e"])
    df["per_y"] = df["a"] ** 1.5
    df["n"] = GAUSS_K_DEG / df["per_y"]
    for c in SIGMA_COLS + ["rms"]:
        if c not in df or df[c].isna().all():
            df[c] = defaults[c]
        df[c] = df[c].fillna(defaults[c])
    if "moid" not in df:
        df["moid"] = np.nan
    missing = df["moid"].isna()
    df["moid_source"] = np.where(missing, "computed", "supplied")
    df.loc[missing, "moid"] = [compute_moid(r.a, r.e, r.i, r.om, r.w)
                               for r in df.loc[missing].itertuples()]
    df["moid_extra_sigma"] = np.where(missing, COMPUTED_MOID_SIGMA_AU, 0.0)
    return df


def score(df: pd.DataFrame, defaults: dict) -> pd.DataFrame:
    df = complete_elements(df, defaults)
    df["p_pha_ml"] = model().predict_proba(build_features(df))[:, 1]
    df = df.join(impact_proxy(df))
    df["hazard_score"] = df["p_pha_ml"] * df["impact_proxy_per_yr"].fillna(0)
    df["moid_ld"] = df["moid"] * 389.17
    return df
