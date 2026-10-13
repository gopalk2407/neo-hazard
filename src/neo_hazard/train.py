"""Train a gradient-boosted classifier for P(PHA | orbital elements, H, uncertainties)."""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

from .data import ROOT, load
from .features import build_features

MODEL_PATH = ROOT / "models" / "pha_hgb.joblib"
METRICS_PATH = ROOT / "outputs" / "metrics.json"


def train(seed: int = 42) -> dict:
    df = load()
    labeled = df[df["pha"].notna()]
    X = build_features(labeled)
    y = labeled["pha"].astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    model = HistGradientBoostingClassifier(
        max_iter=1000, learning_rate=0.05, max_leaf_nodes=63, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=50,
        scoring="average_precision", random_state=seed,
    )
    model.fit(X_tr, y_tr)

    p = model.predict_proba(X_te)[:, 1]
    metrics = {
        "n_train": len(X_tr), "n_test": len(X_te), "test_positives": int(y_te.sum()),
        "roc_auc": roc_auc_score(y_te, p),
        "pr_auc": average_precision_score(y_te, p),
        "n_iter": model.n_iter_,
        "report@0.5": classification_report(y_te, p >= 0.5, output_dict=True)["1"],
    }
    # Permutation importance on a positive-enriched test subsample (full test set is slow).
    sub = pd.concat([X_te[y_te == 1], X_te[y_te == 0].sample(20000, random_state=seed)])
    pi = permutation_importance(model, sub, y_te.loc[sub.index], scoring="average_precision",
                                n_repeats=3, random_state=seed)
    imp = pd.Series(pi.importances_mean, index=X.columns).clip(lower=0)
    metrics["top_features_perm"] = (imp / imp.sum()).sort_values(ascending=False).head(10).round(4).to_dict()

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, default=float))
    return metrics


if __name__ == "__main__":
    print(json.dumps(train(), indent=2, default=float))
