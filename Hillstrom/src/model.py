from __future__ import annotations

from typing import Any, Dict, List, Sequence
import copy

import numpy as np

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

try:
    from catboost import CatBoostClassifier
except ImportError:  # pragma: no cover
    CatBoostClassifier = None


# Centralized default CatBoost hyperparameters for all probability models.
CATBOOST_DEFAULT_PARAMS: Dict[str, Any] = {
    "loss_function": "Logloss",
    "eval_metric": "Logloss",
    "iterations": 500,
    "depth": 6,
    "learning_rate": 0.03,
    "random_seed": 0,
    "verbose": False,
    "allow_writing_files": False,
}


def _require_catboost() -> None:
    if CatBoostClassifier is None:
        raise ImportError(
            "catboost is required for the Hillstrom models. "
            "Please install it with `pip install catboost`."
        )
    if pd is None:
        raise ImportError("pandas is required for catboost-based hillstrom models.")


def _normalize_prob_vector(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float).reshape(-1)
    if p.size == 0:
        return p
    p = np.where(np.isfinite(p), p, 0.0)
    p = np.maximum(p, 0.0)
    s = float(p.sum())
    if s <= 0.0:
        return np.full(p.shape[0], 1.0 / p.shape[0], dtype=float)
    return p / s


def _merge_catboost_params(user_params: Dict[str, Any] | None) -> Dict[str, Any]:
    params = copy.deepcopy(CATBOOST_DEFAULT_PARAMS)
    if user_params:
        params.update(user_params)
    return params


def _to_2d_object_array(X: Any) -> np.ndarray:
    X_arr = np.asarray(X, dtype=object)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(1, -1)
    if X_arr.ndim != 2:
        raise ValueError(f"X must be 2D or 1D row-like, got shape={X_arr.shape}.")
    return X_arr


def _infer_categorical_feature_names(X_df: "pd.DataFrame") -> List[str]:
    categorical: List[str] = []
    for col in X_df.columns:
        s = X_df[col]
        s_non_na = s.dropna()
        if s_non_na.shape[0] == 0:
            categorical.append(str(col))
            continue
        numeric_try = pd.to_numeric(s_non_na, errors="coerce")
        if numeric_try.isna().any():
            categorical.append(str(col))
    return categorical


class _CatBoostProbMixin:
    def __init__(
        self,
        feature_names: Sequence[str] | None = None,
        categorical_feature_names: Sequence[str] | None = None,
        catboost_params: Dict[str, Any] | None = None,
    ):
        _require_catboost()
        self.feature_names: List[str] | None = None if feature_names is None else [str(c) for c in feature_names]
        self.categorical_feature_names: List[str] | None = (
            None if categorical_feature_names is None else [str(c) for c in categorical_feature_names]
        )
        self.catboost_params = _merge_catboost_params(catboost_params)

    def _feature_names_for_width(self, width: int) -> List[str]:
        if self.feature_names is not None:
            if len(self.feature_names) != width:
                raise ValueError(
                    f"feature_names length mismatch: expected {width}, got {len(self.feature_names)}."
                )
            return list(self.feature_names)
        return [f"x{i}" for i in range(width)]

    def _to_catboost_frame(self, X: Any) -> "pd.DataFrame":
        if pd is None:
            raise ImportError("pandas is required for catboost-based models.")
        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
        else:
            X_arr = _to_2d_object_array(X)
            cols = self._feature_names_for_width(X_arr.shape[1])
            X_df = pd.DataFrame(X_arr, columns=cols)

        if self.feature_names is not None:
            missing_cols = [c for c in self.feature_names if c not in X_df.columns]
            if missing_cols:
                raise ValueError(f"X is missing required feature columns: {missing_cols}")
            X_df = X_df.loc[:, self.feature_names].copy()
        else:
            self.feature_names = [str(c) for c in X_df.columns]
            X_df.columns = self.feature_names

        if self.categorical_feature_names is None:
            self.categorical_feature_names = _infer_categorical_feature_names(X_df)
        missing_cat = [c for c in self.categorical_feature_names if c not in X_df.columns]
        if missing_cat:
            raise ValueError(f"categorical_feature_names contains missing columns: {missing_cat}")

        cat_set = set(self.categorical_feature_names)
        for col in X_df.columns:
            if col in cat_set:
                # Keep categorical columns as string for native CatBoost handling.
                X_df[col] = X_df[col].astype("string").fillna("__NA__").astype(str)
            else:
                X_df[col] = pd.to_numeric(X_df[col], errors="coerce")

        return X_df

    def _cat_features_for_fit(self) -> List[str]:
        return [] if self.categorical_feature_names is None else list(self.categorical_feature_names)


class CatBoostOutcomeModel(_CatBoostProbMixin):
    """
    Estimates P_hat(Y | X, A) by fitting one CatBoost classifier per action.
    Falls back to empirical probabilities when an action subset has too few classes.
    """

    def __init__(
        self,
        action_set: Sequence[Any],
        label_set: Sequence[Any],
        feature_names: Sequence[str] | None = None,
        categorical_feature_names: Sequence[str] | None = None,
        catboost_params: Dict[str, Any] | None = None,
    ):
        super().__init__(
            feature_names=feature_names,
            categorical_feature_names=categorical_feature_names,
            catboost_params=catboost_params,
        )
        self.action_set: List[Any] = list(action_set)
        self.label_set: List[Any] = list(label_set)
        self.label_to_index: Dict[Any, int] = {y: j for j, y in enumerate(self.label_set)}
        self.models: Dict[Any, tuple[str, Any]] = {}

    def fit(self, X: np.ndarray, A: np.ndarray, Y: np.ndarray):
        X_df = self._to_catboost_frame(X)
        A = np.asarray(A).reshape(-1)
        Y = np.asarray(Y).reshape(-1)

        if X_df.shape[0] != A.shape[0] or X_df.shape[0] != Y.shape[0]:
            raise ValueError("X, A, Y must have the same number of rows.")

        cat_features = self._cat_features_for_fit()
        for a in self.action_set:
            mask = (A == a)
            X_a = X_df.loc[mask, :]
            Y_a = Y[mask]

            if X_a.shape[0] == 0:
                probs = np.full(len(self.label_set), 1.0 / len(self.label_set), dtype=float)
                self.models[a] = ("empirical", probs)
                continue

            unique_labels = np.unique(Y_a)
            if unique_labels.size == 1:
                probs = np.zeros(len(self.label_set), dtype=float)
                probs[self.label_to_index[unique_labels[0]]] = 1.0
                self.models[a] = ("empirical", probs)
                continue

            clf = CatBoostClassifier(**self.catboost_params)
            clf.fit(X_a, Y_a, cat_features=cat_features)
            self.models[a] = ("catboost", clf)
        return self

    def _predict_from_one_model(self, X_any: Any, a: Any) -> np.ndarray:
        if a not in self.models:
            raise ValueError(f"Action {a} has no fitted model.")
        kind, model = self.models[a]
        X_df = self._to_catboost_frame(X_any)
        n = X_df.shape[0]

        if kind == "empirical":
            probs = np.asarray(model, dtype=float)
            return np.tile(probs, (n, 1))

        clf: CatBoostClassifier = model
        proba = np.asarray(clf.predict_proba(X_df), dtype=float)
        if proba.ndim == 1:
            proba = proba.reshape(-1, 1)

        out = np.zeros((n, len(self.label_set)), dtype=float)
        classes = np.asarray(clf.classes_)
        for cls_idx, y_val in enumerate(classes):
            if y_val in self.label_to_index:
                out[:, self.label_to_index[y_val]] = proba[:, cls_idx]
        out = np.asarray([_normalize_prob_vector(row) for row in out], dtype=float)
        return out

    def predict_proba(self, x: np.ndarray, a: Any) -> np.ndarray:
        return self._predict_from_one_model(x, a)[0]

    def predict_proba_batch(self, X: np.ndarray, a: Any) -> np.ndarray:
        return self._predict_from_one_model(X, a)

    def predict_proba_all_actions(self, x: np.ndarray) -> Dict[Any, np.ndarray]:
        return {a: self.predict_proba(x, a) for a in self.action_set}


class OutcomeModel(CatBoostOutcomeModel):
    """Backward-compatible name: now backed by CatBoost."""


class CatBoostMarginalLabelModel(_CatBoostProbMixin):
    """Estimates P_hat(Y|X) with CatBoost."""

    def __init__(
        self,
        label_set: Sequence[Any],
        feature_names: Sequence[str] | None = None,
        categorical_feature_names: Sequence[str] | None = None,
        catboost_params: Dict[str, Any] | None = None,
    ):
        super().__init__(
            feature_names=feature_names,
            categorical_feature_names=categorical_feature_names,
            catboost_params=catboost_params,
        )
        self.label_set: List[Any] = list(label_set)
        self.label_to_index: Dict[Any, int] = {y: j for j, y in enumerate(self.label_set)}
        self.kind: str | None = None
        self.empirical_probs: np.ndarray | None = None
        self.clf: CatBoostClassifier | None = None

    def fit(self, X: np.ndarray, Y: np.ndarray):
        X_df = self._to_catboost_frame(X)
        Y = np.asarray(Y).reshape(-1)
        if X_df.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must have the same number of rows.")

        unique_labels = np.unique(Y)
        if unique_labels.size <= 1:
            probs = np.zeros(len(self.label_set), dtype=float)
            if unique_labels.size == 0:
                probs[:] = 1.0 / len(self.label_set)
            else:
                probs[self.label_to_index[unique_labels[0]]] = 1.0
            self.kind = "empirical"
            self.empirical_probs = probs
            self.clf = None
            return self

        clf = CatBoostClassifier(**self.catboost_params)
        clf.fit(X_df, Y, cat_features=self._cat_features_for_fit())
        self.kind = "catboost"
        self.empirical_probs = None
        self.clf = clf
        return self

    def predict_proba_batch(self, X: np.ndarray) -> np.ndarray:
        X_df = self._to_catboost_frame(X)
        n = X_df.shape[0]
        if self.kind is None:
            raise RuntimeError("MarginalLabelModel not fit yet.")
        if self.kind == "empirical":
            probs = np.asarray(self.empirical_probs, dtype=float)
            return np.tile(probs, (n, 1))

        if self.clf is None:
            raise RuntimeError("MarginalLabelModel has no trained CatBoost model.")
        raw = np.asarray(self.clf.predict_proba(X_df), dtype=float)
        if raw.ndim == 1:
            raw = raw.reshape(-1, 1)
        out = np.zeros((n, len(self.label_set)), dtype=float)
        classes = np.asarray(self.clf.classes_)
        for cls_idx, y_val in enumerate(classes):
            if y_val in self.label_to_index:
                out[:, self.label_to_index[y_val]] = raw[:, cls_idx]
        out = np.asarray([_normalize_prob_vector(row) for row in out], dtype=float)
        return out

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.predict_proba_batch(x)[0]


class MarginalLabelModel(CatBoostMarginalLabelModel):
    """Backward-compatible name: now backed by CatBoost."""


class BehaviorModel:
    """Fixed Hillstrom behavior policy pi(A|X): P(A=0)=1/3, P(A=1)=2/3."""

    def __init__(self, action_set: Sequence[Any]):
        self.action_set: List[Any] = list(action_set)
        self.action_to_index: Dict[Any, int] = {a: i for i, a in enumerate(self.action_set)}
        if len(self.action_set) != 2 or set(self.action_set) != {0, 1}:
            raise ValueError(
                "BehaviorModel fixed policy only supports action_set exactly equal to {0, 1}; "
                f"got {self.action_set}."
            )
        self._fixed_action_proba = np.zeros(len(self.action_set), dtype=float)
        self._fixed_action_proba[self.action_to_index[0]] = 1.0 / 3.0
        self._fixed_action_proba[self.action_to_index[1]] = 2.0 / 3.0
        self._is_fit = False

    def fit(self, X: np.ndarray, A: np.ndarray):
        X = np.asarray(X, dtype=object)
        A = np.asarray(A)
        if A.ndim != 1:
            A = A.reshape(-1)
        if X.shape[0] != A.shape[0]:
            raise ValueError(
                f"X and A must have the same number of samples, got {X.shape[0]} and {A.shape[0]}."
            )
        observed_actions = set(np.unique(A).tolist())
        unknown_actions = observed_actions.difference(set(self.action_set))
        if unknown_actions:
            raise ValueError(
                "A contains actions not in action_set: "
                f"{sorted(list(unknown_actions))} not in {self.action_set}."
            )
        self._is_fit = True
        return self

    def predict_action_proba(self, x: np.ndarray) -> np.ndarray:
        if not self._is_fit:
            raise RuntimeError("BehaviorModel not fit yet. Call fit() first.")
        return self._fixed_action_proba.copy()

    def predict_action_proba_batch(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fit:
            raise RuntimeError("BehaviorModel not fit yet. Call fit() first.")
        X = np.asarray(X, dtype=object)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return np.tile(self._fixed_action_proba, (X.shape[0], 1))


class CatBoostBehaviorModel(_CatBoostProbMixin):
    """Optional learned behavior model P(A|X) with CatBoost."""

    def __init__(
        self,
        action_set: Sequence[Any],
        feature_names: Sequence[str] | None = None,
        categorical_feature_names: Sequence[str] | None = None,
        catboost_params: Dict[str, Any] | None = None,
    ):
        super().__init__(
            feature_names=feature_names,
            categorical_feature_names=categorical_feature_names,
            catboost_params=catboost_params,
        )
        self.action_set: List[Any] = list(action_set)
        self.action_to_index: Dict[Any, int] = {a: i for i, a in enumerate(self.action_set)}
        self.kind: str | None = None
        self.empirical_probs: np.ndarray | None = None
        self.clf: CatBoostClassifier | None = None

    def fit(self, X: np.ndarray, A: np.ndarray):
        X_df = self._to_catboost_frame(X)
        A = np.asarray(A).reshape(-1)
        if X_df.shape[0] != A.shape[0]:
            raise ValueError("X and A must have the same number of rows.")

        unique_actions = np.unique(A)
        if unique_actions.size <= 1:
            probs = np.zeros(len(self.action_set), dtype=float)
            if unique_actions.size == 0:
                probs[:] = 1.0 / len(self.action_set)
            else:
                probs[self.action_to_index[unique_actions[0]]] = 1.0
            self.kind = "empirical"
            self.empirical_probs = probs
            self.clf = None
            return self

        clf = CatBoostClassifier(**self.catboost_params)
        clf.fit(X_df, A, cat_features=self._cat_features_for_fit())
        self.kind = "catboost"
        self.empirical_probs = None
        self.clf = clf
        return self

    def predict_action_proba_batch(self, X: np.ndarray) -> np.ndarray:
        X_df = self._to_catboost_frame(X)
        n = X_df.shape[0]
        if self.kind is None:
            raise RuntimeError("CatBoostBehaviorModel not fit yet.")
        if self.kind == "empirical":
            probs = np.asarray(self.empirical_probs, dtype=float)
            return np.tile(probs, (n, 1))

        if self.clf is None:
            raise RuntimeError("CatBoostBehaviorModel has no trained CatBoost model.")
        raw = np.asarray(self.clf.predict_proba(X_df), dtype=float)
        if raw.ndim == 1:
            raw = raw.reshape(-1, 1)
        out = np.zeros((n, len(self.action_set)), dtype=float)
        classes = np.asarray(self.clf.classes_)
        for cls_idx, a_val in enumerate(classes):
            if a_val in self.action_to_index:
                out[:, self.action_to_index[a_val]] = raw[:, cls_idx]
        out = np.asarray([_normalize_prob_vector(row) for row in out], dtype=float)
        return out

    def predict_action_proba(self, x: np.ndarray) -> np.ndarray:
        return self.predict_action_proba_batch(x)[0]
