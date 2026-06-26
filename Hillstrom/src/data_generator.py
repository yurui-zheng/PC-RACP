from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence, Tuple
import os
import urllib.request

import numpy as np

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

from sklearn.preprocessing import StandardScaler

UtilityFunc = Callable[[Any, Any], float]

# The same URL used by scikit-uplift to fetch the dataset.
HILLSTROM_URL = "https://hillstorm1.s3.us-east-2.amazonaws.com/hillstorm_no_indices.csv.gz"

_DEFAULT_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "pc_racp_conformal_policy")
_DEFAULT_CACHE_PATH = os.path.join(_DEFAULT_CACHE_DIR, "hillstrom_no_indices.csv.gz")

# Fixed covariates used throughout the Hillstrom experiments.
FIXED_COVARIATE_COLS = ["recency", "history", "mens", "womens", "newbie"]


def _download_if_missing(url: str, dest_path: str) -> None:
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if not os.path.isfile(dest_path):
        print(f"[data] Downloading Hillstrom dataset to: {dest_path}")
        urllib.request.urlretrieve(url, dest_path)


def load_hillstrom_dataset(
    cache_path: str = _DEFAULT_CACHE_PATH,
    standardize: bool = False,
    covariate_cols: list[str] | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load Hillstrom and preprocess into (X, Y, A).

    - A: segment in {No E-Mail, Mens E-Mail, Womens E-Mail} mapped to {0,1},
         where Mens/Womens are merged into one "Email" action.
    - Y: binary visit in {0,1} (0 = no visit, 1 = visit).
    - X: fixed covariates [recency, history, mens, womens, newbie]
         with original categorical columns preserved
         (no one-hot encoding), optionally standardized on numeric columns only.
    """
    if pd is None:
        raise ImportError("pandas is required to load the Hillstrom dataset. Please `pip install pandas`.")

    _download_if_missing(HILLSTROM_URL, cache_path)

    df = pd.read_csv(cache_path, compression="gzip")
    required = {"segment", "visit"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Hillstrom CSV missing required columns: {sorted(list(missing))}")

    # --- action A ---
    seg_map = {"No E-Mail": 0, "Mens E-Mail": 1, "Womens E-Mail": 1}
    if not set(df["segment"].unique()).issubset(set(seg_map.keys())):
        unknown = sorted(list(set(df["segment"].unique()) - set(seg_map.keys())))
        raise ValueError(f"Unknown segment values in dataset: {unknown}")
    A = df["segment"].map(seg_map).astype(int).to_numpy()

    # --- label Y (binary visit) ---
    visit = np.rint(df["visit"].astype(float).to_numpy()).astype(int)
    visit = np.clip(visit, 0, 1)
    Y = visit.astype(int)

    # --- features X ---
    if covariate_cols is not None and list(covariate_cols) != list(FIXED_COVARIATE_COLS):
        raise ValueError(
            "Hillstrom experiments use fixed covariates only: "
            f"{FIXED_COVARIATE_COLS}. Received: {list(covariate_cols)}"
        )

    feature_cols = list(FIXED_COVARIATE_COLS)
    duplicate_features = sorted({c for c in feature_cols if feature_cols.count(c) > 1})
    if duplicate_features:
        raise ValueError(f"Duplicate covariate names are not allowed: {duplicate_features}")

    missing_features = [c for c in feature_cols if c not in df.columns]
    if missing_features:
        raise ValueError(
            "Requested covariates are missing from Hillstrom dataset: "
            f"{sorted(missing_features)}. Available columns: {list(df.columns)}"
        )
    if len(feature_cols) == 0:
        raise ValueError("No covariate columns selected. Please provide at least one valid column.")

    X_df = df.loc[:, feature_cols].copy()

    categorical_feature_names: list[str] = []
    numeric_feature_names: list[str] = []
    for col in feature_cols:
        dtype = X_df[col].dtype
        if (
            pd.api.types.is_object_dtype(dtype)
            or pd.api.types.is_string_dtype(dtype)
            or pd.api.types.is_categorical_dtype(dtype)
        ):
            # Keep native categorical columns as string/category for CatBoost.
            X_df[col] = X_df[col].astype("string").fillna("__NA__").astype(str)
            categorical_feature_names.append(col)
        else:
            numeric_feature_names.append(col)

    if standardize:
        # Optional legacy path: only scale numeric features.
        if len(numeric_feature_names) > 0:
            scaler = StandardScaler(with_mean=True, with_std=True)
            X_df.loc[:, numeric_feature_names] = scaler.fit_transform(X_df[numeric_feature_names].to_numpy(dtype=float))

    X = X_df.to_numpy(dtype=object)
    categorical_feature_indices = [feature_cols.index(c) for c in categorical_feature_names]

    meta: Dict[str, Any] = {
        "dataset": "hillstrom",
        "n_rows": int(df.shape[0]),
        "feature_names": list(feature_cols),
        "covariate_cols_used": list(feature_cols),
        "categorical_feature_names": list(categorical_feature_names),
        "numeric_feature_names": list(numeric_feature_names),
        "categorical_feature_indices": list(categorical_feature_indices),
        "standardize": bool(standardize),
        "action_mapping": seg_map,
        "label_name": "visit_binary",
        "visit_bins": {0: "0", 1: "1"},
        "visit_counts": {int(k): int(v) for k, v in zip(*np.unique(Y, return_counts=True))},
    }
    return X, Y, A, meta


def generate_data(
    n_samples: int = 20000,
    random_state: int | None = 0,
    return_env: bool = False,
    covariate_cols: list[str] | None = None,
    standardize: bool = False,
):
    """Backward-compatible wrapper.

    The previous synthetic generator returned (X, Y, A, env) when return_env=True.
    For real data, returns (X, Y, A, meta) instead, where meta includes
    dataset-specific info (feature names, visit label bins, etc.).
    """
    X_full, Y_full, A_full, meta = load_hillstrom_dataset(
        covariate_cols=covariate_cols,
        standardize=standardize,
    )
    rng = np.random.default_rng(random_state)

    X, Y, A = X_full, Y_full, A_full
    n_rows = X_full.shape[0]
    if n_samples is not None and int(n_samples) > 0 and int(n_samples) < n_rows:
        idx = rng.choice(n_rows, size=int(n_samples), replace=False)
        X = X_full[idx]
        Y = Y_full[idx]
        A = A_full[idx]

    X = np.asarray(X, dtype=object)
    Y = np.asarray(Y, dtype=int)
    A = np.asarray(A, dtype=int)

    meta = {
        **meta,
        "synthetic_bandit": False,
    }

    if return_env:
        return X, Y, A, meta
    return X, Y, A


U_MAX: float = 1.0

# Utility table used in the Hillstrom experiment.
# Rows are actions a=0 (No E-Mail), a=1 (Email); columns are labels y=0, y=1.
_UTILITY_TABLE = np.array([
    [0.40, 0.25],
    [0.10, 0.90],
], dtype=float)


def utility_func(a: Any, y: Any) -> float:
    return float(_UTILITY_TABLE[int(a), int(y)])
