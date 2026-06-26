from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, List, Dict, Any, Tuple
import hashlib
import warnings
import numpy as np

UtilityFunc = Callable[[Any, Any], float]


@dataclass
class PCRACPConformalPolicyConfig:
    alpha: float
    action_set: Sequence[Any]
    label_set: Sequence[Any]
    utility_func: UtilityFunc
    u_max: float
    beta_max: float = 2.0
    min_pi_prob: float = 1e-6
    # Numerical tolerance for comparisons like score >= 0.
    tie_tol: float = 1e-12


class PCRACPConformalPolicy:
    """
    PC-RACP action-dependent conformal policy.
    """

    def __init__(self, config: PCRACPConformalPolicyConfig):
        self.cfg = config
        self.action_set = list(config.action_set)
        self.label_set = list(config.label_set)
        self.utility_func = config.utility_func
        self.alpha = config.alpha
        self.u_max = config.u_max

        self.action_to_index: Dict[Any, int] = {a: i for i, a in enumerate(self.action_set)}
        self.label_to_index: Dict[Any, int] = {y: j for j, y in enumerate(self.label_set)}

        self.U_matrix = self._build_utility_matrix()
        self._utility_sort_order = [np.argsort(self.U_matrix[i]) for i in range(len(self.action_set))]
        self._utility_sorted = [self.U_matrix[i][order] for i, order in enumerate(self._utility_sort_order)]
        self.beta_hat: float | None = None
        self._candidate_cache: Dict[bytes, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def _build_utility_matrix(self) -> np.ndarray:
        m = len(self.action_set)
        n = len(self.label_set)
        U = np.zeros((m, n), dtype=float)
        for i, a in enumerate(self.action_set):
            for j, y in enumerate(self.label_set):
                U[i, j] = float(self.utility_func(a, y))
        return U

    # ---------- quantile (discrete) ----------
    @staticmethod
    def _quantile_precomputed(u_sorted: np.ndarray, cdf: np.ndarray, alpha: float) -> float:
        """
        Discrete "upper" quantile (right-continuous generalized inverse):

            quant^+_alpha(Z) := sup{ z in R : P(Z <= z) <= alpha }
                             = inf{ z in R : P(Z <= z) >  alpha }.

        This version takes the "high value" at jump points: if alpha equals a CDF jump value,
        it returns the next support value (i.e., the value right after the jump).

        u_sorted must be sorted ascending and cdf = cumulative probs along u_sorted.
        """
        u_sorted = np.asarray(u_sorted, dtype=float).reshape(-1)
        cdf = np.asarray(cdf, dtype=float).reshape(-1)
        a = float(alpha)
        eps = 1e-12

        # Clamp extremes (keep behavior well-defined / numerically stable)
        if a <= 0.0 + eps:
            return float(u_sorted[0])
        if a >= 1.0 - eps:
            return float(u_sorted[-1])

        # First index where cdf[idx] > a  (strictly greater) -> "high value at jumps"
        idx = int(np.searchsorted(cdf, a + eps, side="right"))
        idx = min(max(idx, 0), u_sorted.size - 1)
        return float(u_sorted[idx])

    def _prepare_action_distributions_for_x(self, x: np.ndarray, outcome_model: Any):
        action_info: List[Dict[str, np.ndarray]] = []
        candidate_alphas = [0.0, 1.0]
        probs_by_action = None
        if hasattr(outcome_model, "predict_proba_all_actions"):
            probs_by_action = outcome_model.predict_proba_all_actions(x)

        for a_idx, a in enumerate(self.action_set):
            if probs_by_action is None:
                prob_y = np.asarray(outcome_model.predict_proba(x, a), dtype=float)
            else:
                prob_y = np.asarray(probs_by_action[a], dtype=float)
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
            order = self._utility_sort_order[a_idx]
            u_sorted = self._utility_sorted[a_idx]
            p_sorted = prob_y[order]
            cdf = np.cumsum(p_sorted)
            action_info.append({"u_sorted": u_sorted, "cdf": cdf})
            candidate_alphas.extend(cdf.tolist())

        alpha_candidates = np.unique(np.clip(candidate_alphas, 0.0, 1.0))
        return action_info, alpha_candidates

    @staticmethod
    def _cache_key_for_x(x: np.ndarray) -> bytes:
        x_arr = np.asarray(x)
        if x_arr.dtype == object:
            flat_vals = np.asarray(x_arr, dtype=object).reshape(-1).tolist()
            serial_parts = []
            for v in flat_vals:
                if isinstance(v, float) and np.isnan(v):
                    serial_parts.append("__nan__")
                else:
                    serial_parts.append(repr(v))
            payload = "|".join(serial_parts).encode("utf-8")
            digest = hashlib.sha1(payload).digest()
            return (x_arr.shape, "object", digest.hex()).__repr__().encode("utf-8")

        x_contig = np.ascontiguousarray(x_arr)
        digest = hashlib.sha1(x_contig.tobytes(order="C")).digest()
        return (x_contig.shape, x_contig.dtype.str, digest).__repr__().encode("utf-8")

    def _theta_s_candidates_for_x(
            self, x: np.ndarray, outcome_model: Any
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        cache_key = self._cache_key_for_x(x)
        cached = self._candidate_cache.get(cache_key)
        if cached is not None:
            return cached

        action_info, alpha_candidates = self._prepare_action_distributions_for_x(x, outcome_model)

        s_list: List[float] = []
        theta_list: List[float] = []
        aidx_list: List[int] = []

        for alpha in alpha_candidates:
            s = 1.0 - float(alpha)
            if s < -1e-12 or s > 1.0 + 1e-12:
                continue

            if np.isclose(s, 0.0):
                theta = float(self.u_max)
                a_idx_star = int(np.argmax(self.U_matrix.max(axis=1)))
            else:
                gamma_vals = []
                for info in action_info:
                    gamma_a = self._quantile_precomputed(info["u_sorted"], info["cdf"], float(alpha))
                    gamma_vals.append(gamma_a)

                gamma_vals = np.asarray(gamma_vals, dtype=float)
                theta = float(gamma_vals.max())
                a_idx_star = int(gamma_vals.argmax())

            s_list.append(float(s))
            theta_list.append(theta)
            aidx_list.append(a_idx_star)

        s_c = np.asarray(s_list, dtype=float)
        theta_c = np.asarray(theta_list, dtype=float)
        aidx_c = np.asarray(aidx_list, dtype=int)
        self._candidate_cache[cache_key] = (s_c, theta_c, aidx_c)
        return s_c, theta_c, aidx_c

    def _compute_g_for_x(self, x: np.ndarray, beta: float, outcome_model: Any) -> Tuple[float, float, Any]:
        s_c, theta_c, aidx_c = self._theta_s_candidates_for_x(x, outcome_model)
        obj = theta_c + float(beta) * s_c
        k_star = int(np.argmax(obj))
        a_star = self.action_set[int(aidx_c[k_star])]
        return float(s_c[k_star]), float(theta_c[k_star]), a_star

    # ---------- learn beta by bisection ----------
    def _mean_coverage_for_beta(self, X_learn: np.ndarray, outcome_model: Any, beta: float) -> float:
        s_values = [self._compute_g_for_x(x, beta, outcome_model)[0] for x in X_learn]
        return float(np.mean(s_values)) if len(s_values) > 0 else 0.0

    def learn_beta(self, X_learn: np.ndarray, outcome_model: Any) -> float:
        target = 1.0 - self.alpha
        beta_lo = 0.0
        beta_hi = float(self.cfg.beta_max) if float(self.cfg.beta_max) > 0.0 else 1.0

        cov_lo = self._mean_coverage_for_beta(X_learn, outcome_model, beta_lo)
        if cov_lo >= target:
            self.beta_hat = beta_lo
            return self.beta_hat

        cov_hi = self._mean_coverage_for_beta(X_learn, outcome_model, beta_hi)

        if cov_hi < target:
            hi_cap = 1e6
            for _ in range(40):
                if beta_hi >= hi_cap:
                    break
                beta_hi *= 2.0
                cov_hi = self._mean_coverage_for_beta(X_learn, outcome_model, beta_hi)
                if cov_hi >= target:
                    break

        if cov_hi < target:
            self.beta_hat = float(beta_hi)
            return self.beta_hat

        for _ in range(40):
            beta_mid = 0.5 * (beta_lo + beta_hi)
            cov_mid = self._mean_coverage_for_beta(X_learn, outcome_model, beta_mid)
            if cov_mid >= target:
                beta_hi = beta_mid
            else:
                beta_lo = beta_mid

        self.beta_hat = float(beta_hi)
        return self.beta_hat

    # ---------- calibration scores ----------
    def _policy_action_for_x(self, x: np.ndarray, outcome_model: Any, beta: float) -> Tuple[float, float, Any]:
        return self._compute_g_for_x(x, beta, outcome_model)

    @staticmethod
    def _theta_star_from_candidates(
            s_candidates: np.ndarray, theta_candidates: np.ndarray, beta: float
    ) -> float:
        obj = theta_candidates + float(beta) * s_candidates
        return float(theta_candidates[int(np.argmax(obj))])

    @staticmethod
    def _theta_star_from_candidate_matrix(
            s_mat: np.ndarray, theta_mat: np.ndarray, mask: np.ndarray, beta: float
    ) -> np.ndarray:
        if s_mat.size == 0:
            return np.zeros(0, dtype=float)
        obj = theta_mat + float(beta) * s_mat
        obj = np.where(mask, obj, -np.inf)
        k_star = np.argmax(obj, axis=1)
        return theta_mat[np.arange(theta_mat.shape[0]), k_star]

    def _build_candidate_matrix(
            self, X: np.ndarray, outcome_model: Any
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if X.size == 0:
            return np.zeros((0, 0), dtype=float), np.zeros((0, 0), dtype=float), np.zeros((0, 0), dtype=bool)
        s_list: List[np.ndarray] = []
        theta_list: List[np.ndarray] = []
        max_len = 0
        for x in X:
            s_c, theta_c, _ = self._theta_s_candidates_for_x(x, outcome_model)
            s_list.append(s_c)
            theta_list.append(theta_c)
            max_len = max(max_len, s_c.size)
        n = len(s_list)
        s_mat = np.zeros((n, max_len), dtype=float)
        theta_mat = np.zeros((n, max_len), dtype=float)
        mask = np.zeros((n, max_len), dtype=bool)
        for i, (s_c, theta_c) in enumerate(zip(s_list, theta_list)):
            length = s_c.size
            if length == 0:
                continue
            s_mat[i, :length] = s_c
            theta_mat[i, :length] = theta_c
            mask[i, :length] = True
        return s_mat, theta_mat, mask

    def _beta_star_for_label(
            self,
            u_test: float,
            s_test: np.ndarray,
            theta_test: np.ndarray,
            u_cal: np.ndarray,
            w_cal: np.ndarray,
            w_test: float,
            s_cal: np.ndarray,
            theta_cal: np.ndarray,
            cal_mask: np.ndarray,
            cache_theta_test: Dict[float, float],
            cache_cal_num: Dict[float, float],
    ) -> float:
        total_w = float(w_cal.sum()) + float(w_test)
        target = 1.0 - self.alpha
        if total_w <= 0.0:
            return float(max(self.cfg.beta_max, 1.0))

        def cal_num_for_beta(beta: float) -> float:
            cached = cache_cal_num.get(beta)
            if cached is not None:
                return cached
            theta_star_cal = self._theta_star_from_candidate_matrix(s_cal, theta_cal, cal_mask, beta)
            scores_cal = u_cal - theta_star_cal
            num = float(w_cal[scores_cal >= 0.0].sum())
            cache_cal_num[beta] = num
            return num

        def theta_test_for_beta(beta: float) -> float:
            cached = cache_theta_test.get(beta)
            if cached is not None:
                return cached
            theta_val = self._theta_star_from_candidates(s_test, theta_test, beta)
            cache_theta_test[beta] = theta_val
            return theta_val

        def val_at_beta(beta: float) -> float:
            score_test = u_test - theta_test_for_beta(beta)
            num = cal_num_for_beta(beta)
            if score_test >= 0.0:
                num += float(w_test)
            return num / total_w

        beta_lo = 0.0
        beta_hi = max(float(self.beta_hat) if self.beta_hat is not None else 0.0, 1.0)
        beta_cap = max(float(self.cfg.beta_max), 10)

        if val_at_beta(beta_lo) >= target:
            return beta_lo

        while val_at_beta(beta_hi) < target and beta_hi < beta_cap:
            beta_hi *= 2.0
            if beta_hi > beta_cap:
                beta_hi = beta_cap
                break

        if val_at_beta(beta_hi) < target:
            return float(beta_cap)

        for _ in range(50):
            beta_mid = 0.5 * (beta_lo + beta_hi)
            if val_at_beta(beta_mid) >= target:
                beta_hi = beta_mid
            else:
                beta_lo = beta_mid
        return float(beta_hi)

    # ---------- prediction set for one x ----------
    def _prediction_set_for_single_x(
            self,
            x: np.ndarray,
            cal_payload: Dict[str, np.ndarray],
            outcome_model: Any,
            behavior_model: Any,
            beta_hat: float,
            cache_cal_num: Dict[float, float],
            pi_vec_test: np.ndarray | None = None,
    ):
        u_cal = cal_payload["u_cal"]
        w_cal = cal_payload["w_cal"]
        s_cal = cal_payload["s_cal"]
        theta_cal = cal_payload["theta_cal"]
        cal_mask = cal_payload["cal_mask"]

        _, _, a_x = self._policy_action_for_x(x, outcome_model, beta_hat)
        a_idx = self.action_to_index[a_x]

        if pi_vec_test is None:
            pi_vec = np.asarray(behavior_model.predict_action_proba(x), dtype=float)
        else:
            pi_vec = np.asarray(pi_vec_test, dtype=float)
        pi_a = max(float(pi_vec[a_idx]), self.cfg.min_pi_prob)
        w_test = 1.0 / pi_a

        C_x: List[Any] = []
        s_test, theta_test, _ = self._theta_s_candidates_for_x(x, outcome_model)
        cache_theta_test: Dict[float, float] = {}

        for y in self.label_set:
            y_idx = self.label_to_index[y]
            u_test = float(self.U_matrix[a_idx, y_idx])
            beta_star = self._beta_star_for_label(
                u_test=u_test,
                s_test=s_test,
                theta_test=theta_test,
                u_cal=u_cal,
                w_cal=w_cal,
                w_test=w_test,
                s_cal=s_cal,
                theta_cal=theta_cal,
                cal_mask=cal_mask,
                cache_theta_test=cache_theta_test,
                cache_cal_num=cache_cal_num,
            )
            theta_star = cache_theta_test.get(beta_star)
            if theta_star is None:
                theta_star = self._theta_star_from_candidates(s_test, theta_test, beta_star)
                cache_theta_test[beta_star] = theta_star
            score_test = u_test - theta_star
            if score_test + float(self.cfg.tie_tol) >= 0.0:
                C_x.append(y)

        u_x = self.u_max if len(C_x) == 0 else float(min(self.utility_func(a_x, y) for y in C_x))
        return C_x, u_x, a_x

    def build_prediction_sets(
            self,
            X_cal: np.ndarray,
            Y_cal: np.ndarray,
            A_cal: np.ndarray,
            X_test: np.ndarray,
            outcome_model: Any,
            behavior_model: Any,
    ):
        if self.beta_hat is None:
            raise RuntimeError("Call learn_beta(...) first.")

        u_cal_list = []
        w_cal_list = []
        x_cal_list = []

        pi_cal_matrix = None
        if hasattr(behavior_model, "predict_action_proba_batch"):
            pi_cal_matrix = np.asarray(behavior_model.predict_action_proba_batch(X_cal), dtype=float)
        pi_test_matrix = None
        if hasattr(behavior_model, "predict_action_proba_batch"):
            pi_test_matrix = np.asarray(behavior_model.predict_action_proba_batch(X_test), dtype=float)

        for idx, (x, y, a_obs) in enumerate(zip(X_cal, Y_cal, A_cal)):
            _, _, a_hat = self._policy_action_for_x(x, outcome_model, self.beta_hat)
            if a_obs != a_hat:
                continue

            a_idx = self.action_to_index[a_obs]
            y_idx = self.label_to_index[y]
            u_cal_list.append(float(self.U_matrix[a_idx, y_idx]))

            if pi_cal_matrix is None:
                pi_vec = np.asarray(behavior_model.predict_action_proba(x), dtype=float)
            else:
                pi_vec = np.asarray(pi_cal_matrix[idx], dtype=float)
            pi_a = max(float(pi_vec[a_idx]), self.cfg.min_pi_prob)
            w_cal_list.append(float(1.0 / pi_a))
            x_cal_list.append(np.asarray(x, dtype=object))

        if len(u_cal_list) == 0:
            warnings.warn(
                "PC-RACP calibration set is empty after action filtering; "
                "falling back to empty calibration scores.",
                RuntimeWarning,
            )

        if len(u_cal_list) == 0:
            u_cal = np.zeros(0, dtype=float)
            w_cal = np.zeros(0, dtype=float)
            n_features = int(np.asarray(X_cal).shape[1]) if np.asarray(X_cal).ndim >= 2 else 0
            x_cal = np.empty((0, n_features), dtype=object)
            s_cal = np.zeros((0, 0), dtype=float)
            theta_cal = np.zeros((0, 0), dtype=float)
            cal_mask = np.zeros((0, 0), dtype=bool)
        else:
            u_cal = np.asarray(u_cal_list, dtype=float)
            w_cal = np.asarray(w_cal_list, dtype=float)
            x_cal = np.asarray(x_cal_list, dtype=object)
            s_cal, theta_cal, cal_mask = self._build_candidate_matrix(x_cal, outcome_model)

        cal_payload = {
            "u_cal": u_cal,
            "w_cal": w_cal,
            "s_cal": s_cal,
            "theta_cal": theta_cal,
            "cal_mask": cal_mask,
        }

        sets = []
        actions = []
        certs = []
        cache_cal_num_shared: Dict[float, float] = {}
        for idx, x in enumerate(X_test):
            C_x, u_x, a_x = self._prediction_set_for_single_x(
                x,
                cal_payload,
                outcome_model,
                behavior_model,
                self.beta_hat,
                cache_cal_num_shared,
                pi_vec_test=None if pi_test_matrix is None else pi_test_matrix[idx],
            )
            sets.append(C_x)
            actions.append(a_x)
            certs.append(u_x)

        stats = {
            "beta_hat": float(self.beta_hat),
            "cert_avg": float(np.mean(certs)) if len(certs) else float("nan"),
            "avg_set_size": float(np.mean([len(s) for s in sets])) if len(sets) else float("nan"),
        }
        return sets, actions, stats

