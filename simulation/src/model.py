from __future__ import annotations

from typing import Any, Dict, List
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

SUPPORTED_MODEL_TYPES = ("logistic", "random_forest")
RF_DEFAULT_N_ESTIMATORS = 200
RF_DEFAULT_MAX_DEPTH = 14
RF_DEFAULT_MIN_SAMPLES_LEAF = 2
RF_DEFAULT_N_JOBS = 1


def _align_proba_to_index(
    proba: np.ndarray,
    classes: np.ndarray,
    index_map: Dict[Any, int],
    size: int,
    fill_value: float = 0.0,
) -> np.ndarray:
    res = np.full(size, fill_value, dtype=float)
    for cls_idx, label_val in enumerate(classes):
        j = index_map[label_val]
        res[j] = proba[cls_idx]
    s = res.sum()
    if s <= 0:
        res[:] = 1.0 / size
    else:
        res /= s
    return res


def _align_proba_matrix_to_index(
    proba: np.ndarray,
    classes: np.ndarray,
    index_map: Dict[Any, int],
    size: int,
    fill_value: float = 0.0,
) -> np.ndarray:
    proba = np.asarray(proba, dtype=float)
    if proba.ndim != 2:
        raise ValueError("proba must be a 2D array.")

    res = np.full((proba.shape[0], size), fill_value, dtype=float)
    for cls_idx, label_val in enumerate(classes):
        j = index_map[label_val]
        res[:, j] = proba[:, cls_idx]

    row_sums = res.sum(axis=1)
    bad = row_sums <= 0
    if np.any(bad):
        res[bad] = 1.0 / size
        row_sums = res.sum(axis=1)
    res = res / row_sums[:, None]
    return res


def _validate_model_type(model_type: str) -> str:
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"Invalid model_type={model_type!r}. "
            f"Expected one of {list(SUPPORTED_MODEL_TYPES)}."
        )
    return model_type


def _make_classifier(
    model_type: str,
    random_state: int,
    n_estimators: int,
    max_depth: int | None,
    min_samples_leaf: int,
    n_jobs: int,
):
    mt = _validate_model_type(model_type)
    if mt == "logistic":
        # Keep existing logistic defaults unchanged for backward compatibility.
        return LogisticRegression(max_iter=500)
    if mt == "random_forest":
        return RandomForestClassifier(
            n_estimators=int(n_estimators),
            max_depth=max_depth,
            min_samples_leaf=int(min_samples_leaf),
            random_state=int(random_state),
            n_jobs=int(n_jobs),
        )
    raise ValueError(
        f"Invalid model_type={model_type!r}. "
        f"Expected one of {list(SUPPORTED_MODEL_TYPES)}."
    )


def _as_2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim == 1:
        return X.reshape(1, -1)
    if X.ndim != 2:
        raise ValueError("X must be a 1D or 2D array.")
    return X


def _uniform_probs(size: int) -> np.ndarray:
    probs = np.ones(int(size), dtype=float)
    probs /= probs.sum()
    return probs


def _one_hot_probs(value: Any, index_map: Dict[Any, int], size: int) -> np.ndarray:
    probs = np.zeros(int(size), dtype=float)
    probs[index_map[value]] = 1.0
    return probs


class OutcomeModel:
    """
    Estimates P_hat(Y | X, A) by fitting a separate classifier per action.
    If an action has too little data / only one class, falls back to an empirical distribution.
    """

    def __init__(
        self,
        action_set,
        label_set,
        model_type: str = "logistic",
        random_state: int = 0,
        n_estimators: int = RF_DEFAULT_N_ESTIMATORS,
        max_depth: int | None = RF_DEFAULT_MAX_DEPTH,
        min_samples_leaf: int = RF_DEFAULT_MIN_SAMPLES_LEAF,
        n_jobs: int = RF_DEFAULT_N_JOBS,
    ):
        self.action_set: List[Any] = list(action_set)
        self.label_set: List[Any] = list(label_set)
        self.action_to_index: Dict[Any, int] = {a: i for i, a in enumerate(self.action_set)}
        self.label_to_index: Dict[Any, int] = {y: j for j, y in enumerate(self.label_set)}
        self.model_type = _validate_model_type(model_type)
        self.random_state = int(random_state)
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.min_samples_leaf = int(min_samples_leaf)
        self.n_jobs = int(n_jobs)
        self.models: Dict[Any, Any] = {}

    def _new_classifier(self):
        return _make_classifier(
            model_type=self.model_type,
            random_state=self.random_state,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            n_jobs=self.n_jobs,
        )

    def _get_model(self, a: Any):
        if not self.models:
            raise RuntimeError("OutcomeModel not fit yet. Call fit() first.")
        if a not in self.models:
            raise ValueError(f"Unknown action {a!r}. Expected one of {self.action_set}.")
        return self.models[a]

    def fit(self, X: np.ndarray, A: np.ndarray, Y: np.ndarray):
        X = _as_2d(X)
        A = np.asarray(A)
        Y = np.asarray(Y)
        if X.shape[0] != A.shape[0] or X.shape[0] != Y.shape[0]:
            raise ValueError("X, A, Y must have the same number of rows.")

        self.models = {}

        for a in self.action_set:
            mask = (A == a)
            X_a = X[mask]
            Y_a = Y[mask]

            if X_a.shape[0] == 0:
                probs = _uniform_probs(len(self.label_set))
                self.models[a] = ("empirical", probs)
                continue

            unique_labels = np.unique(Y_a)
            if unique_labels.size == 1:
                probs = _one_hot_probs(unique_labels[0], self.label_to_index, len(self.label_set))
                self.models[a] = ("empirical", probs)
                continue

            clf = self._new_classifier()
            clf.fit(X_a, Y_a)
            self.models[a] = (self.model_type, clf)

    def predict_proba(self, x: np.ndarray, a: Any) -> np.ndarray:
        kind, model = self._get_model(a)
        if kind == "empirical":
            return np.asarray(model, dtype=float)

        clf = model
        x = _as_2d(x)
        proba = clf.predict_proba(x)[0]
        return _align_proba_to_index(
            proba,
            clf.classes_,
            self.label_to_index,
            len(self.label_set),
            fill_value=0.0,
        )

    def predict_proba_batch(self, X: np.ndarray, a: Any) -> np.ndarray:
        kind, model = self._get_model(a)
        X = _as_2d(X)

        if kind == "empirical":
            probs = np.asarray(model, dtype=float).reshape(1, -1)
            return np.repeat(probs, X.shape[0], axis=0)

        clf = model
        proba = clf.predict_proba(X)
        return _align_proba_matrix_to_index(
            proba,
            clf.classes_,
            self.label_to_index,
            len(self.label_set),
            fill_value=0.0,
        )

    def predict_proba_all_actions(self, x: np.ndarray) -> Dict[Any, np.ndarray]:
        return {a: self.predict_proba(x, a) for a in self.action_set}

    def predict_proba_all_actions_batch(self, X: np.ndarray) -> Dict[Any, np.ndarray]:
        """
        Batch version of predict_proba_all_actions.

        Returns:
            Dict[action, probs], where probs has shape (n_samples, n_labels).
        """
        X = _as_2d(X)
        return {a: self.predict_proba_batch(X, a) for a in self.action_set}


class BehaviorModel:
    """Estimates pi_hat(A|X) with a configurable classifier backend."""

    def __init__(
        self,
        action_set,
        model_type: str = "logistic",
        random_state: int = 0,
        n_estimators: int = RF_DEFAULT_N_ESTIMATORS,
        max_depth: int | None = RF_DEFAULT_MAX_DEPTH,
        min_samples_leaf: int = RF_DEFAULT_MIN_SAMPLES_LEAF,
        n_jobs: int = RF_DEFAULT_N_JOBS,
    ):
        self.action_set: List[Any] = list(action_set)
        self.action_to_index: Dict[Any, int] = {a: i for i, a in enumerate(self.action_set)}
        self.model_type = _validate_model_type(model_type)
        self.random_state = int(random_state)
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.min_samples_leaf = int(min_samples_leaf)
        self.n_jobs = int(n_jobs)
        self.clf: Any | None = None
        self._fallback_probs: np.ndarray | None = None

    def _new_classifier(self):
        return _make_classifier(
            model_type=self.model_type,
            random_state=self.random_state,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            n_jobs=self.n_jobs,
        )

    def fit(self, X: np.ndarray, A: np.ndarray):
        X = _as_2d(X)
        A = np.asarray(A)
        if X.shape[0] != A.shape[0]:
            raise ValueError("X and A must have the same number of rows.")
        if A.shape[0] == 0:
            raise ValueError("BehaviorModel.fit received empty training data.")

        unique_actions = np.unique(A)
        if unique_actions.size == 1:
            self._fallback_probs = _one_hot_probs(
                unique_actions[0],
                self.action_to_index,
                len(self.action_set),
            )
            self.clf = None
            return

        clf = self._new_classifier()
        clf.fit(X, A)
        self.clf = clf
        self._fallback_probs = None

    def predict_action_proba(self, x: np.ndarray) -> np.ndarray:
        if self._fallback_probs is not None:
            return np.asarray(self._fallback_probs, dtype=float)
        if self.clf is None:
            raise RuntimeError("BehaviorModel not fit yet. Call fit() first.")

        x = _as_2d(x)
        proba = self.clf.predict_proba(x)[0]
        return _align_proba_to_index(
            proba,
            self.clf.classes_,
            self.action_to_index,
            len(self.action_set),
            fill_value=1e-8,
        )

    def predict_action_proba_batch(self, X: np.ndarray) -> np.ndarray:
        X = _as_2d(X)
        if self._fallback_probs is not None:
            probs = np.asarray(self._fallback_probs, dtype=float).reshape(1, -1)
            return np.repeat(probs, X.shape[0], axis=0)
        if self.clf is None:
            raise RuntimeError("BehaviorModel not fit yet. Call fit() first.")

        proba = self.clf.predict_proba(X)
        return _align_proba_matrix_to_index(
            proba,
            self.clf.classes_,
            self.action_to_index,
            len(self.action_set),
            fill_value=1e-8,
        )


class MarginalLabelModel:
    """
    Estimates P_hat(Y|X) ignoring actions (for RAC baseline).
    """

    def __init__(
        self,
        label_set,
        model_type: str = "logistic",
        random_state: int = 0,
        n_estimators: int = RF_DEFAULT_N_ESTIMATORS,
        max_depth: int | None = RF_DEFAULT_MAX_DEPTH,
        min_samples_leaf: int = RF_DEFAULT_MIN_SAMPLES_LEAF,
        n_jobs: int = RF_DEFAULT_N_JOBS,
    ):
        self.label_set: List[Any] = list(label_set)
        self.label_to_index: Dict[Any, int] = {y: j for j, y in enumerate(self.label_set)}
        self.model_type = _validate_model_type(model_type)
        self.random_state = int(random_state)
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.min_samples_leaf = int(min_samples_leaf)
        self.n_jobs = int(n_jobs)
        self.clf: Any | None = None
        self._fallback_probs: np.ndarray | None = None

    def _new_classifier(self):
        return _make_classifier(
            model_type=self.model_type,
            random_state=self.random_state,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            n_jobs=self.n_jobs,
        )

    def fit(self, X: np.ndarray, Y: np.ndarray):
        X = _as_2d(X)
        Y = np.asarray(Y)
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must have the same number of rows.")
        if Y.shape[0] == 0:
            raise ValueError("MarginalLabelModel.fit received empty training data.")

        unique_labels = np.unique(Y)
        if unique_labels.size == 1:
            self._fallback_probs = _one_hot_probs(
                unique_labels[0],
                self.label_to_index,
                len(self.label_set),
            )
            self.clf = None
            return

        clf = self._new_classifier()
        clf.fit(X, Y)
        self.clf = clf
        self._fallback_probs = None

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self._fallback_probs is not None:
            return np.asarray(self._fallback_probs, dtype=float)
        if self.clf is None:
            raise RuntimeError("MarginalLabelModel not fit yet.")
        x = _as_2d(x)
        proba = self.clf.predict_proba(x)[0]
        return _align_proba_to_index(
            proba,
            self.clf.classes_,
            self.label_to_index,
            len(self.label_set),
            fill_value=0.0,
        )

    def predict_proba_batch(self, X: np.ndarray) -> np.ndarray:
        X = _as_2d(X)
        if self._fallback_probs is not None:
            probs = np.asarray(self._fallback_probs, dtype=float).reshape(1, -1)
            return np.repeat(probs, X.shape[0], axis=0)
        if self.clf is None:
            raise RuntimeError("MarginalLabelModel not fit yet.")

        proba = self.clf.predict_proba(X)
        return _align_proba_matrix_to_index(
            proba,
            self.clf.classes_,
            self.label_to_index,
            len(self.label_set),
            fill_value=0.0,
        )
