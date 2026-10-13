"""Load the JPL Small-Body Database export and cache it as parquet."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "dataset.csv"
CACHE_PATH = ROOT / "data" / "sbdb.parquet"

# Identifier / text columns; everything else numeric except the flags below.
ID_COLS = ["id", "spkid", "full_name", "pdes", "name", "prefix", "orbit_id",
           "equinox", "epoch_cal", "tp_cal", "class"]
FLAG_COLS = ["neo", "pha"]


def load(csv_path: Path = CSV_PATH, cache_path: Path = CACHE_PATH) -> pd.DataFrame:
    if cache_path.exists() and cache_path.stat().st_mtime >= csv_path.stat().st_mtime:
        return pd.read_parquet(cache_path)

    df = pd.read_csv(csv_path, low_memory=False, dtype={c: "string" for c in ID_COLS})
    df["full_name"] = df["full_name"].str.strip()
    for c in FLAG_COLS:
        df[c] = df[c].map({"Y": 1, "N": 0}).astype("Int8")
    num_cols = [c for c in df.columns if c not in ID_COLS + FLAG_COLS]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    return df
