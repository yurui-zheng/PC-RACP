from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Callable, Tuple

import numpy as np

UtilityFunc = Callable[[Any, Any], float]


@dataclass
class RACConfig:
    """
    Risk-Averse Calibration (RAC) baseline - **strictly following Algorithm 1** in the paper.

    Key point (vs. "split" simplifications):
      - For each candidate label y in Y, compute a label-dependent beta beta_hat_y(x_test)
        by (bracket +) bisection on a 1D beta axis such that the *augmented* empirical
        coverage constraint holds:

            ( sum_{i=1}^n  1{ Y_i in C_hat(X_i; beta) }  +  1{ y in C_hat(x_test; beta) } ) / (n+1)  >=  1 - alpha

      - The final RAC set is:
            C_RAC(x_test) = { y : y in C_hat(x_test; beta_hat_y(x_test)) }.

    This implementation assumes:
      - X is continuous/numeric, Y is discrete finite, A is discrete finite.
      - RAC uses a marginal label model f(x) ~= P(Y|X=x) (ignoring actions),
        consistent with the baseline used in the paper.

    Interface notes:
      - fit(X_cal, Y_cal, f_model) caches calibration candidates.
      - predict_set(x_test) returns (C, nu_star, a_star, beta_summary)
        where beta_summary is a diagnostic scalar (mean of beta_hat_y over y).
    """

    alpha: float
    action_set: Sequence[Any]
    label_set: Sequence[Any]
    utility_func: UtilityFunc
    u_max: float

    # Search range for beta; expand upper bracket automatically if needed.
    beta_min: float = -50.0
    beta_max: float = 50.0

    # Bisection settings
    bisect_iters: int = 60
    bisect_tol: float = 1e-7

    # Numerical tolerance used in the membership test u(a_hat, y) >= theta - eps
    eps: float = 0.0

    # Max number of times to expand beta_max if it is insufficient.
    bracket_expand_max: int = 50


class RiskAverseCalibration:
    def __init__(self, cfg: RACConfig):
        self.cfg = cfg
        self.alpha = float(cfg.alpha)
        self.action_set = list(cfg.action_set)
        self.label_set = list(cfg.label_set)
        self.utility_func = cfg.utility_func
        self.u_max = float(cfg.u_max)

        self.beta_min = float(cfg.beta_min)
        self.beta_max = float(cfg.beta_max)
        self.bisect_iters = int(cfg.bisect_iters)
        self.bisect_tol = float(cfg.bisect_tol)
        self.eps = float(cfg.eps)
        self.bracket_expand_max = int(cfg.bracket_expand_max)

        # label/action indexing
        self.label_to_index: Dict[Any, int] = {y: j for j, y in enumerate(self.label_set)}
        self.action_to_index: Dict[Any, int] = {a: i for i, a in enumerate(self.action_set)}

        # Precompute utility matrix U[a_idx, y_idx]
        self.U_matrix = np.zeros((len(self.action_set), len(self.label_set)), dtype=float)
        for i, a in enumerate(self.action_set):
            for j, y in enumerate(self.label_set):
                self.U_matrix[i, j] = float(self.utility_func(a, y))

        # Cached after fit()
        self._f_model: Any | None = None
        self._n_cal: int | None = None
        self._cal_S: np.ndarray | None = None        # (n, Kmax)
        self._cal_Theta: np.ndarray | None = None    # (n, Kmax)
        self._cal_Aidx: np.ndarray | None = None     # (n, Kmax) action index for each candidate
        self._cal_mask: np.ndarray | None = None     # (n, Kmax) candidate validity mask
        self._cal_Utrue: np.ndarray | None = None    # (n, Kmax) u(a_hat_candidate, y_true)
        self._cal_idx: np.ndarray | None = None
        self._last_beta_by_label: Dict[Any, float] | None = None
        self._cal_hit_cache: Dict[float, int] = {}

        # Backward-compat scalar (NOT used by Algorithm 1; kept only if other code accesses it)
        self._beta_hat_diag: float | None = None

    # ========= internal primitives =========
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
    def _theta_s_candidates_for_x(self, prob_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        For a given x represented via prob_y = f(x) over labels,
        enumerate candidate triples (s, theta(s), a_hat(s)) induced by discrete quantiles.

        Returns:
          s_c     : (K,) candidate s values, where s = 1 - alpha_candidate
          theta_c : (K,) theta(s) = max_a quantile_alpha[u(a,Y)]
          aidx_c  : (K,) argmax action index achieving theta(s)
        """
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

        candidate_alphas: List[float] = [0.0, 1.0]
        action_infos: List[Tuple[np.ndarray, np.ndarray]] = []

        # For each action, sort utilities and compute CDF of utilities under prob_y
        for a_idx in range(len(self.action_set)):
            u_vals = self.U_matrix[a_idx]  # (n_labels,)
            order = np.argsort(u_vals)
            u_sorted = u_vals[order]
            p_sorted = prob_y[order]
            cdf = np.cumsum(p_sorted)
            action_infos.append((u_sorted, cdf))
            candidate_alphas.extend(cdf.tolist())

        alpha_candidates = np.unique(np.clip(candidate_alphas, 0.0, 1.0))

        s_list: List[float] = []
        theta_list: List[float] = []
        aidx_list: List[int] = []

        for alpha in alpha_candidates:
            s = 1.0 - float(alpha)
            if np.isclose(s, 0.0):
                theta = float(self.u_max)
                aidx = int(np.argmax(self.U_matrix.max(axis=1)))
            else:
                gamma_vals = []
                for (u_sorted, cdf) in action_infos:
                    gamma_vals.append(self._quantile_precomputed(u_sorted, cdf, float(alpha)))
                gamma_vals = np.asarray(gamma_vals, dtype=float)
                theta = float(gamma_vals.max())
                aidx = int(gamma_vals.argmax())

            s_list.append(float(s))
            theta_list.append(theta)
            aidx_list.append(aidx)

        return (
            np.asarray(s_list, dtype=float),
            np.asarray(theta_list, dtype=float),
            np.asarray(aidx_list, dtype=int),
        )

    # ========= calibration cache and fast coverage computation =========
    def _prepare_calibration_cache(self, X_cal: np.ndarray, Y_cal: np.ndarray, f_model: Any):
        """
        Precompute (S, Theta, Aidx, mask, Utrue) on the calibration set.
        These allow O(n*Kmax) hitcount evaluation for any beta.
        """
        self._f_model = f_model
        X_cal = np.asarray(X_cal)
        Y_cal = np.asarray(Y_cal)

        n_cal = int(X_cal.shape[0])
        self._n_cal = n_cal

        y_idx = np.array([self.label_to_index[y] for y in Y_cal], dtype=int)

        # First pass: compute variable-length candidate arrays, track maximum K
        all_s: List[np.ndarray] = []
        all_theta: List[np.ndarray] = []
        all_aidx: List[np.ndarray] = []
        Kmax = 0

        probs_cal = None
        if hasattr(f_model, "predict_proba_batch"):
            probs_cal = np.asarray(f_model.predict_proba_batch(X_cal), dtype=float)
            if probs_cal.shape[0] != n_cal:
                raise ValueError("predict_proba_batch returned wrong number of rows.")

        for i in range(n_cal):
            prob_y = np.asarray(
                probs_cal[i] if probs_cal is not None else f_model.predict_proba(X_cal[i]), dtype=float
            )
            s_c, theta_c, aidx_c = self._theta_s_candidates_for_x(prob_y)
            all_s.append(s_c)
            all_theta.append(theta_c)
            all_aidx.append(aidx_c)
            Kmax = max(Kmax, int(s_c.size))

        # Allocate padded arrays
        S = np.zeros((n_cal, Kmax), dtype=float)
        Theta = np.full((n_cal, Kmax), -np.inf, dtype=float)
        Aidx = np.zeros((n_cal, Kmax), dtype=int)
        Utrue = np.full((n_cal, Kmax), -np.inf, dtype=float)
        mask = np.zeros((n_cal, Kmax), dtype=bool)

        for i in range(n_cal):
            s_c = all_s[i]
            theta_c = all_theta[i]
            aidx_c = all_aidx[i]
            k = int(s_c.size)

            S[i, :k] = s_c
            Theta[i, :k] = theta_c
            Aidx[i, :k] = aidx_c
            mask[i, :k] = True

            yi = int(y_idx[i])
            # u(a_hat_candidate, y_true) for each candidate
            Utrue[i, :k] = self.U_matrix[aidx_c[:k], yi]

        self._cal_S = S
        self._cal_Theta = Theta
        self._cal_Aidx = Aidx
        self._cal_mask = mask
        self._cal_Utrue = Utrue
        self._cal_idx = np.arange(n_cal, dtype=int)
        self._cal_hit_cache = {}

    @staticmethod
    def _beta_cache_key(beta: float) -> float:
        return float(round(float(beta), 12))

    def _calibration_hitcount_for_beta(self, beta: float) -> int:
        """
        Return sum_i 1{Y_i in C_hat(X_i; beta)} on the calibration set.
        Uses cached (S,Theta,mask,Utrue).
        """
        if (
            self._cal_S is None
            or self._cal_Theta is None
            or self._cal_mask is None
            or self._cal_Utrue is None
            or self._cal_idx is None
        ):
            raise RuntimeError("Calibration cache not prepared. Call fit() first.")

        S = self._cal_S
        Theta = self._cal_Theta
        mask = self._cal_mask
        Utrue = self._cal_Utrue
        cal_idx = self._cal_idx

        obj = Theta + float(beta) * S
        obj = np.where(mask, obj, -np.inf)
        k_star = np.argmax(obj, axis=1)  # (n,)

        theta_star = Theta[cal_idx, k_star]
        u_true_star = Utrue[cal_idx, k_star]
        hits = u_true_star >= (theta_star - self.eps)
        return int(np.sum(hits))

    # ========= public API =========
    def fit(self, X_cal: np.ndarray, Y_cal: np.ndarray, f_model: Any):
        """
        Algorithm 1 RAC uses *no* single global beta; fit only caches calibration candidates.
        """
        self._prepare_calibration_cache(X_cal, Y_cal, f_model)
        self._beta_hat_diag = float("nan")
        self._last_beta_by_label = None
        self._cal_hit_cache = {}

    def predict_set(self, x_test: np.ndarray) -> Tuple[List[Any], float, Any, float]:
        """
        Algorithm 1:
          - For each y in Y, find beta_hat_y(x_test) by bisection to satisfy the (n+1)-augmented
            coverage constraint.
          - Include y in output set iff y in C_hat(x_test; beta_hat_y).
        Returns:
          C, nu_star, a_star, beta_summary
        """
        if self._f_model is None or self._n_cal is None:
            raise RuntimeError("Call fit(X_cal, Y_cal, f_model) first.")

        x_test = np.asarray(x_test)
        if x_test.ndim == 2 and x_test.shape[0] == 1:
            x_in = x_test[0]
        elif x_test.ndim == 1:
            x_in = x_test
        else:
            raise ValueError("x_test must be shape (d,) or (1,d)")

        prob_y = np.asarray(self._f_model.predict_proba(x_in), dtype=float)
        return self._predict_set_from_prob(prob_y)

    def predict_set_from_prob(self, prob_y: np.ndarray) -> Tuple[List[Any], float, Any, float]:
        prob_y = np.asarray(prob_y, dtype=float)
        return self._predict_set_from_prob(prob_y)

    def _predict_set_from_prob(self, prob_y: np.ndarray) -> Tuple[List[Any], float, Any, float]:
        s_c, theta_c, aidx_c = self._theta_s_candidates_for_x(prob_y)

        n = int(self._n_cal)
        target = 1.0 - self.alpha
        def test_hit_for_beta(beta: float, y_idx: int) -> bool:
            obj = theta_c + float(beta) * s_c
            k_star = int(np.argmax(obj))
            theta_star = float(theta_c[k_star])
            aidx_star = int(aidx_c[k_star])
            return bool(self.U_matrix[aidx_star, y_idx] >= (theta_star - self.eps))

        def cov_y(beta: float, y_idx: int) -> float:
            beta_key = self._beta_cache_key(beta)
            if beta_key in self._cal_hit_cache:
                cal_hits = self._cal_hit_cache[beta_key]
            else:
                cal_hits = self._calibration_hitcount_for_beta(beta_key)
                self._cal_hit_cache[beta_key] = cal_hits
            hit_test = 1 if test_hit_for_beta(beta, y_idx) else 0
            return float((cal_hits + hit_test) / (n + 1))

        def learn_beta_for_label(y_idx: int) -> float:
            lo = float(self.beta_min)
            hi = float(self.beta_max)

            cov_lo = cov_y(lo, y_idx)
            if cov_lo >= target:
                return lo

            cov_hi = cov_y(hi, y_idx)
            expands = 0
            while cov_hi < target and expands < self.bracket_expand_max:
                span = max(hi - lo, 1.0)
                hi = hi + 2.0 * span
                cov_hi = cov_y(hi, y_idx)
                expands += 1

            if cov_hi < target:
                # Give up: return the largest tried hi (will under-cover); still deterministic.
                return float(hi)

            # Bisection for minimal beta achieving target
            for _ in range(self.bisect_iters):
                mid = 0.5 * (lo + hi)
                cov_mid = cov_y(mid, y_idx)
                if cov_mid >= target:
                    hi = mid
                else:
                    lo = mid
                if abs(hi - lo) <= self.bisect_tol * (1.0 + abs(hi) + abs(lo)):
                    break
            return float(hi)

        beta_by_label: Dict[Any, float] = {}
        in_set: Dict[Any, bool] = {}

        for y in self.label_set:
            y_idx = int(self.label_to_index[y])
            beta_y = float(learn_beta_for_label(y_idx))
            beta_by_label[y] = beta_y
            in_set[y] = bool(test_hit_for_beta(beta_y, y_idx))

        self._last_beta_by_label = dict(beta_by_label)

        C = [y for y in self.label_set if in_set[y]]
        a_star, nu_star = self.max_min_action(C)

        beta_summary = float(np.mean(list(beta_by_label.values()))) if len(beta_by_label) else float("nan")
        return C, nu_star, a_star, beta_summary

    def max_min_action(self, C: List[Any]) -> Tuple[Any, float]:
        """
        v(C) = max_a min_{y in C} u(a,y). If C is empty, return (a0, u_max).
        """
        if len(C) == 0:
            return self.action_set[0], float(self.u_max)

        y_indices = np.array([self.label_to_index[y] for y in C], dtype=int)
        worst = np.min(self.U_matrix[:, y_indices], axis=1)
        a_idx_star = int(np.argmax(worst))
        return self.action_set[a_idx_star], float(worst[a_idx_star])

    @property
    def beta_hat(self) -> float:
        """
        Backward compatibility only. Algorithm 1 does NOT use a single global beta.
        """
        if self._beta_hat_diag is None:
            raise RuntimeError("Call fit(...) first.")
        return float(self._beta_hat_diag)

    @property
    def last_beta_by_label(self) -> Dict[Any, float] | None:
        """
        Debug: beta_y(x_test) computed in the most recent predict_set call.
        """
        return None if self._last_beta_by_label is None else dict(self._last_beta_by_label)
