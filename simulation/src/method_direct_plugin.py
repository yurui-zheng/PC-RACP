from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Callable, Dict, List, Sequence

import numpy as np

UtilityFunc = Callable[[Any, Any], float]


@dataclass
class DirectPluginPolicyConfig:
    alpha: float
    action_set: Sequence[Any]
    label_set: Sequence[Any]
    utility_func: UtilityFunc
    u_max: float
    tie_tol: float = 1e-12


class DirectPluginPolicy:
    """
    Plug-in baseline (Algorithm 3):
      - fixes t = 1 - alpha
      - no beta learning
      - no calibration split
    """

    def __init__(self, config: DirectPluginPolicyConfig):
        self.cfg = config
        self.alpha = float(config.alpha)
        self.action_set = list(config.action_set)
        self.label_set = list(config.label_set)
        self.utility_func = config.utility_func
        self.u_max = float(config.u_max)

        self.action_to_index: Dict[Any, int] = {a: i for i, a in enumerate(self.action_set)}
        self.label_to_index: Dict[Any, int] = {y: j for j, y in enumerate(self.label_set)}

        self.U_matrix = self._build_utility_matrix()
        self._utility_sort_order = [np.argsort(self.U_matrix[i]) for i in range(len(self.action_set))]
        self._utility_sorted = [self.U_matrix[i][order] for i, order in enumerate(self._utility_sort_order)]
        self._candidate_cache: Dict[bytes, np.ndarray] = {}

    def set_alpha(self, alpha: float):
        self.alpha = float(alpha)
        self.cfg.alpha = float(alpha)

    def _build_utility_matrix(self) -> np.ndarray:
        m = len(self.action_set)
        n = len(self.label_set)
        U = np.zeros((m, n), dtype=float)
        for i, a in enumerate(self.action_set):
            for j, y in enumerate(self.label_set):
                U[i, j] = float(self.utility_func(a, y))
        return U

    @staticmethod
    def _quantile_precomputed(u_sorted: np.ndarray, cdf: np.ndarray, alpha: float) -> float:
        """
        Discrete "upper" quantile (right-continuous generalized inverse),
        consistent with existing methods in this repo.
        """
        u_sorted = np.asarray(u_sorted, dtype=float).reshape(-1)
        cdf = np.asarray(cdf, dtype=float).reshape(-1)
        a = float(alpha)
        eps = 1e-12

        if a <= 0.0 + eps:
            return float(u_sorted[0])
        if a >= 1.0 - eps:
            return float(u_sorted[-1])

        idx = int(np.searchsorted(cdf, a + eps, side="right"))
        idx = min(max(idx, 0), u_sorted.size - 1)
        return float(u_sorted[idx])

    def _validate_and_normalize_prob(self, prob_y: np.ndarray) -> np.ndarray:
        prob_y = np.asarray(prob_y, dtype=float).reshape(-1)
        if prob_y.size != len(self.label_set):
            raise ValueError("prob_y must have length = |label_set|")
        if not np.all(np.isfinite(prob_y)):
            raise ValueError("prob_y must contain only finite values")
        if np.any(prob_y < 0.0):
            raise ValueError("prob_y must be non-negative")

        total = float(prob_y.sum())
        if total <= 0.0:
            raise ValueError("prob_y must have positive sum")
        if not np.isclose(total, 1.0, atol=1e-8):
            prob_y = prob_y / total
        return prob_y

    @staticmethod
    def _cache_key_for_x(x: np.ndarray) -> bytes:
        x = np.asarray(x)
        x_contig = np.ascontiguousarray(x)
        digest = hashlib.sha1(x_contig.tobytes(order="C")).digest()
        return (x_contig.shape, x_contig.dtype.str, digest).__repr__().encode()

    @staticmethod
    def _row_probs_from_batch(
        probs_by_action_batch: Dict[Any, np.ndarray],
        action_set: List[Any],
        idx: int,
    ) -> Dict[Any, np.ndarray]:
        return {a: np.asarray(probs_by_action_batch[a][idx], dtype=float) for a in action_set}

    def _cdf_candidates_for_x(
        self,
        x: np.ndarray,
        outcome_model: Any,
        probs_by_action: Dict[Any, np.ndarray] | None = None,
    ) -> np.ndarray:
        cache_key = self._cache_key_for_x(x)
        cached = self._candidate_cache.get(cache_key)
        if cached is not None:
            return cached

        if probs_by_action is None and hasattr(outcome_model, "predict_proba_all_actions"):
            probs_by_action = outcome_model.predict_proba_all_actions(x)

        cdf_mat = np.zeros((len(self.action_set), len(self.label_set)), dtype=float)
        for a_idx, a in enumerate(self.action_set):
            if probs_by_action is None:
                prob_y = np.asarray(outcome_model.predict_proba(x, a), dtype=float)
            else:
                prob_y = np.asarray(probs_by_action[a], dtype=float)
            prob_y = self._validate_and_normalize_prob(prob_y)

            order = self._utility_sort_order[a_idx]
            cdf_mat[a_idx] = np.cumsum(prob_y[order])

        self._candidate_cache[cache_key] = cdf_mat
        return cdf_mat

    def precompute_candidates(
        self,
        X: np.ndarray,
        outcome_model: Any,
        probs_by_action_batch: Dict[Any, np.ndarray] | None = None,
    ):
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array for precompute_candidates.")

        if probs_by_action_batch is not None:
            for a in self.action_set:
                if a not in probs_by_action_batch:
                    raise ValueError(f"Missing action {a!r} in probs_by_action_batch.")
                arr = np.asarray(probs_by_action_batch[a], dtype=float)
                if arr.shape[0] != X.shape[0]:
                    raise ValueError(
                        "Each probs_by_action_batch[a] must have the same number of rows as X."
                    )

        for idx, x in enumerate(X):
            probs_by_action = None
            if probs_by_action_batch is not None:
                probs_by_action = self._row_probs_from_batch(
                    probs_by_action_batch=probs_by_action_batch,
                    action_set=self.action_set,
                    idx=idx,
                )
            self._cdf_candidates_for_x(
                x=x,
                outcome_model=outcome_model,
                probs_by_action=probs_by_action,
            )

    def _prediction_set_for_single_x(
        self,
        x: np.ndarray,
        outcome_model: Any,
        probs_by_action: Dict[Any, np.ndarray] | None = None,
    ):
        cdf_mat = self._cdf_candidates_for_x(
            x=x,
            outcome_model=outcome_model,
            probs_by_action=probs_by_action,
        )

        gamma_vals = np.zeros(len(self.action_set), dtype=float)
        for a_idx, a in enumerate(self.action_set):
            u_sorted = self._utility_sorted[a_idx]
            cdf = cdf_mat[a_idx]
            gamma_vals[a_idx] = self._quantile_precomputed(u_sorted, cdf, self.alpha)

        a_idx_star = int(np.argmax(gamma_vals))
        theta = float(gamma_vals[a_idx_star])
        a_star = self.action_set[a_idx_star]

        in_set = self.U_matrix[a_idx_star] >= (theta - float(self.cfg.tie_tol))
        C_x = [self.label_set[j] for j, keep in enumerate(in_set) if bool(keep)]

        if len(C_x) == 0:
            cert = float(self.u_max)
        else:
            cert = float(np.min(self.U_matrix[a_idx_star, in_set]))
        return C_x, a_star, theta, cert

    def build_prediction_sets(
        self,
        X_test: np.ndarray,
        outcome_model: Any,
        Y_test: np.ndarray | None = None,
        probs_by_action_batch: Dict[Any, np.ndarray] | None = None,
    ):
        if Y_test is not None and len(Y_test) != len(X_test):
            raise ValueError("Y_test must have the same length as X_test.")

        if probs_by_action_batch is not None:
            for a in self.action_set:
                if a not in probs_by_action_batch:
                    raise ValueError(f"Missing action {a!r} in probs_by_action_batch.")
                arr = np.asarray(probs_by_action_batch[a], dtype=float)
                if arr.shape[0] != len(X_test):
                    raise ValueError(
                        "Each probs_by_action_batch[a] must have the same number of rows as X_test."
                    )

        sets: List[List[Any]] = []
        actions: List[Any] = []
        theta_vals: List[float] = []
        cert_vals: List[float] = []
        obs_hits: List[float] = []

        for idx, x in enumerate(X_test):
            probs_by_action = None
            if probs_by_action_batch is not None:
                probs_by_action = self._row_probs_from_batch(
                    probs_by_action_batch=probs_by_action_batch,
                    action_set=self.action_set,
                    idx=idx,
                )

            C_x, a_x, theta_x, cert_x = self._prediction_set_for_single_x(
                x=x,
                outcome_model=outcome_model,
                probs_by_action=probs_by_action,
            )
            sets.append(C_x)
            actions.append(a_x)
            theta_vals.append(theta_x)
            cert_vals.append(cert_x)

            if Y_test is not None:
                obs_hits.append(1.0 if Y_test[idx] in C_x else 0.0)

        stats: Dict[str, float] = {
            "theta_avg": float(np.mean(theta_vals)) if len(theta_vals) else float("nan"),
            "cert_avg": float(np.mean(cert_vals)) if len(cert_vals) else float("nan"),
            "avg_set_size": float(np.mean([len(s) for s in sets])) if len(sets) else float("nan"),
        }
        if Y_test is not None:
            stats["obs_cov_emp"] = float(np.mean(obs_hits)) if len(obs_hits) else float("nan")

        return sets, actions, stats
