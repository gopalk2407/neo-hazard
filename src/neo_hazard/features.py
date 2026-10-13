"""Physics-informed features derived from osculating orbital elements.

MOID and the PHA/NEO flags are deliberately excluded from the model features:
PHA is defined as (MOID <= 0.05 AU and H <= 22), so feeding MOID would let the
model copy the label rule instead of learning close-approach geometry.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

R_EARTH_AU = 6371.0 / 1.495978707e8
V_EARTH_KMS = 29.78
V_ESC_EARTH_KMS = 11.19
# Below ~1 km/s two-body focusing diverges; also catches orbits where the Öpik
# formula has no real solution (U^2 < 0: the orbit cannot cross Earth's).
MIN_ENCOUNTER_KMS = 1.0
A_JUPITER = 5.2026

SIGMA_COLS = ["sigma_e", "sigma_a", "sigma_q", "sigma_i", "sigma_om",
              "sigma_w", "sigma_ma", "sigma_n", "sigma_tp"]


def node_distances(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Heliocentric distance (AU) at the ascending and descending nodes.

    Earth's orbit lies (almost) in the ecliptic, so a near-1 AU node is the
    geometric precondition for an Earth intersection.
    """
    p = df["a"] * (1 - df["e"] ** 2)
    w = np.radians(df["w"])
    return p / (1 + df["e"] * np.cos(w)), p / (1 - df["e"] * np.cos(w))


def opik_encounter_speed(df: pd.DataFrame) -> pd.Series:
    """Öpik unperturbed encounter speed U relative to Earth, in km/s."""
    u2 = 3 - 1 / df["a"] - 2 * np.sqrt(np.clip(df["a"] * (1 - df["e"] ** 2), 0, None)) \
        * np.cos(np.radians(df["i"]))
    return np.sqrt(np.clip(u2, 1e-6, None)) * V_EARTH_KMS


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    for c in ["e", "a", "q", "ad", "i", "om", "w", "n", "per_y", "H", "rms"]:
        X[c] = df[c]
    X["log_a"] = np.log10(df["a"].abs())
    X["sin_i"] = np.sin(np.radians(df["i"]))
    X["cos_w"] = np.cos(np.radians(df["w"]))
    X["sin_w"] = np.sin(np.radians(df["w"]))

    r_asc, r_desc = node_distances(df)
    X["node_asc_minus_1au"] = r_asc - 1
    X["node_desc_minus_1au"] = r_desc - 1
    X["min_node_dist_1au"] = np.minimum((r_asc - 1).abs(), (r_desc - 1).abs())
    X["q_minus_1au"] = df["q"] - 1
    X["Q_minus_1au"] = df["ad"] - 1
    X["crosses_1au"] = ((df["q"] <= 1) & (df["ad"] >= 1)).astype(int)
    X["tisserand_j"] = A_JUPITER / df["a"] + 2 * np.cos(np.radians(df["i"])) \
        * np.sqrt(np.clip(df["a"] / A_JUPITER * (1 - df["e"] ** 2), 0, None))
    X["opik_U_kms"] = opik_encounter_speed(df)

    for c in SIGMA_COLS:
        X[f"log_{c}"] = np.log10(df[c].clip(lower=1e-15))
    return X


def impact_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """Rough, physically motivated proxy for relative collision likelihood.

    P_cross: probability that the *true* MOID is smaller than Earth's
    gravitationally-focused capture radius, treating the catalogue MOID as
    Gaussian with sigma ~ orbit uncertainty in q (floored).
    Annual factor: encounters scale with orbital revolutions per year.
    This is NOT a Sentry-style impact probability (no orbit propagation,
    no resonant returns); it is a ranking heuristic.
    """
    v_inf = opik_encounter_speed(df).clip(lower=MIN_ENCOUNTER_KMS)
    capture_au = R_EARTH_AU * np.sqrt(1 + (V_ESC_EARTH_KMS / v_inf) ** 2)
    sigma = np.sqrt(df["sigma_q"].fillna(1e-2) ** 2 + (df["a"] * df["sigma_e"].fillna(1e-3)) ** 2)
    if "moid_extra_sigma" in df:  # e.g. error of a locally computed MOID
        sigma = np.sqrt(sigma ** 2 + df["moid_extra_sigma"].fillna(0) ** 2)
    sigma = sigma.clip(lower=1e-6)
    m = df["moid"]
    p_cross = norm.cdf((capture_au - m) / sigma) - norm.cdf((-capture_au - m) / sigma)
    out = pd.DataFrame(index=df.index)
    out["capture_radius_au"] = capture_au
    out["moid_sigma_au"] = sigma
    out["p_moid_within_capture"] = p_cross
    out["impact_proxy_per_yr"] = p_cross / df["per_y"].clip(lower=1e-3)
    return out
