from __future__ import annotations

import argparse
import csv
import hashlib
import os
from typing import Any, Dict, List

import numpy as np

from src.data_generator import U_MAX, generate_data, utility_func
from src.method_pc_racp import PCRACPConformalPolicy, PCRACPConformalPolicyConfig
from src.method_rac import RACConfig, RiskAverseCalibration
from src.model import BehaviorModel, MarginalLabelModel, OutcomeModel


DEFAULT_N_SAMPLES = 64000
DEFAULT_ALPHAS = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]
DEFAULT_COVARIATE_COLS = ["recency", "history", "mens", "womens", "newbie"]
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "hillstrom")

PLOT_STYLE = {
    "figsize": (3.4, 2.5),
    "wide_figsize": (6.8, 2.5),
    "title_size": 9,
    "label_size": 8,
    "tick_size": 7,
    "legend_size": 7,
    "line_width": 1.25,
    "marker_size": 3.0,
    "spine_width": 0.75,
    "grid_width": 0.6,
    "grid_color": "#D6D6D6",
    "spine_color": "#A8A8A8",
    "legend_edge_color": "#D0D0D0",
    "dpi": 300,
}

PLOT_FILENAMES = {
    "utility_png": "utility_vs_alpha.png",
    "utility_pdf": "utility_vs_alpha.pdf",
    "coverage_png": "coverage_vs_alpha.png",
    "coverage_pdf": "coverage_vs_alpha.pdf",
    "action1_count_png": "optimal_action1_count_vs_alpha.png",
    "action1_count_pdf": "optimal_action1_count_vs_alpha.pdf",
    "pair_counts_a0_png": "pair_counts_a0_vs_alpha.png",
    "pair_counts_a0_pdf": "pair_counts_a0_vs_alpha.pdf",
    "pair_counts_a1_png": "pair_counts_a1_vs_alpha.png",
    "pair_counts_a1_pdf": "pair_counts_a1_vs_alpha.pdf",
    "a0_set01_rate_png": "set01_rate_given_optimal_a0_vs_alpha.png",
    "a0_set01_rate_pdf": "set01_rate_given_optimal_a0_vs_alpha.pdf",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Hillstrom PC-RACP experiment.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_samples", type=int, default=DEFAULT_N_SAMPLES)
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--alphas",
        type=str,
        default=",".join(f"{a:.2f}" for a in DEFAULT_ALPHAS),
        help="Comma-separated alpha grid, for example: 0.02,0.04,0.06",
    )
    parser.add_argument("--catboost_iterations", type=int, default=None)
    parser.add_argument("--catboost_depth", type=int, default=None)
    parser.add_argument("--catboost_learning_rate", type=float, default=None)
    return parser.parse_args()


def parse_alpha_grid(alpha_text: str) -> List[float]:
    alphas = [float(x.strip()) for x in alpha_text.split(",") if x.strip()]
    if not alphas:
        raise ValueError("Alpha grid is empty.")
    if any(a <= 0.0 or a >= 1.0 for a in alphas):
        raise ValueError("All alpha values must lie in (0, 1).")
    return alphas


def catboost_params_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    params: Dict[str, Any] = {"random_seed": int(args.seed)}
    if args.catboost_iterations is not None:
        params["iterations"] = int(args.catboost_iterations)
    if args.catboost_depth is not None:
        params["depth"] = int(args.catboost_depth)
    if args.catboost_learning_rate is not None:
        params["learning_rate"] = float(args.catboost_learning_rate)
    return params


def set_paper_style(plt) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": PLOT_STYLE["spine_color"],
            "axes.titlesize": PLOT_STYLE["title_size"],
            "axes.labelsize": PLOT_STYLE["label_size"],
            "xtick.labelsize": PLOT_STYLE["tick_size"],
            "ytick.labelsize": PLOT_STYLE["tick_size"],
            "legend.fontsize": PLOT_STYLE["legend_size"],
        }
    )


def style_axis(ax, title: str, y_label: str) -> None:
    ax.set_title(title, fontsize=PLOT_STYLE["title_size"])
    ax.set_xlabel(r"$\alpha$", fontsize=PLOT_STYLE["label_size"])
    ax.set_ylabel(y_label, fontsize=PLOT_STYLE["label_size"])
    ax.tick_params(
        axis="both",
        which="both",
        labelsize=PLOT_STYLE["tick_size"],
        width=PLOT_STYLE["spine_width"],
        length=3.0,
    )
    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=PLOT_STYLE["grid_width"],
        color=PLOT_STYLE["grid_color"],
        alpha=0.9,
    )
    ax.grid(axis="x", visible=False)
    for spine in ax.spines.values():
        spine.set_linewidth(PLOT_STYLE["spine_width"])
        spine.set_color(PLOT_STYLE["spine_color"])
    ax.margins(x=0.02)


def add_legend(ax, ncol: int = 1) -> None:
    ax.legend(
        loc="best",
        ncol=ncol,
        frameon=True,
        framealpha=0.85,
        edgecolor=PLOT_STYLE["legend_edge_color"],
        facecolor="white",
    )


def split_data(
    X: np.ndarray,
    Y: np.ndarray,
    A: np.ndarray,
    train_frac: float = 0.3,
    learn_frac: float = 0.2,
    calib_frac: float = 0.2,
    random_state: int | None = 0,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(random_state)
    n = X.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)

    n_train = int(train_frac * n)
    n_learn = int(learn_frac * n)
    n_calib = int(calib_frac * n)

    used = n_train + n_learn + n_calib
    if used > n:
        raise ValueError("train_frac + learn_frac + calib_frac is too large.")
    if n_calib <= 0:
        raise ValueError("calib_frac is too small; need at least one calibration sample.")

    train_idx = idx[:n_train]
    learn_idx = idx[n_train : n_train + n_learn]
    cal_idx = idx[n_train + n_learn : n_train + n_learn + n_calib]
    test_idx = idx[used:]

    all_idx = np.concatenate([train_idx, learn_idx, cal_idx, test_idx])
    if all_idx.size != n or np.unique(all_idx).size != n:
        raise ValueError("Split indices are not a disjoint partition of the data.")

    return {
        "X_train": X[train_idx],
        "Y_train": Y[train_idx],
        "A_train": A[train_idx],
        "X_learn": X[learn_idx],
        "Y_learn": Y[learn_idx],
        "A_learn": A[learn_idx],
        "X_cal": X[cal_idx],
        "Y_cal": Y[cal_idx],
        "A_cal": A[cal_idx],
        "X_test": X[test_idx],
        "Y_test": Y[test_idx],
        "A_test": A[test_idx],
    }


def fixed_action_worst_value(C_x: List[Any], a: Any, u_func, u_max: float) -> float:
    if C_x is None or len(C_x) == 0:
        return float(u_max)
    return float(min(float(u_func(a, y)) for y in C_x))


def set_maxmin_value(C: List[Any], u_func, u_max: float, action_set: List[Any]) -> float:
    if C is None or len(C) == 0:
        return float(u_max)
    return float(max(min(float(u_func(a, y)) for y in C) for a in action_set))


def _canonicalize_prediction_set(C: Any) -> tuple[int, ...] | None:
    if C is None:
        return ()
    if isinstance(C, np.ndarray):
        items = np.asarray(C).reshape(-1).tolist()
    elif isinstance(C, (list, tuple, set)):
        items = list(C)
    else:
        try:
            items = list(C)
        except TypeError:
            items = [C]

    if len(items) == 0:
        return ()
    try:
        return tuple(sorted(set(int(v) for v in items)))
    except (TypeError, ValueError):
        return None


def _count_action_set_pairs(sets: List[Any], actions: List[Any], n_total: int) -> Dict[str, float]:
    if len(sets) != len(actions):
        raise ValueError("sets and actions must have the same length.")

    counts: Dict[str, int] = {}
    for a in [0, 1]:
        for set_key in ["empty", "0", "1", "01", "other"]:
            counts[f"a{a}_{set_key}_count"] = 0
    counts["other_action_count"] = 0

    for C, a in zip(sets, actions):
        C_key = _canonicalize_prediction_set(C)
        try:
            a_key = int(a)
        except (TypeError, ValueError):
            a_key = None

        if a_key not in (0, 1):
            counts["other_action_count"] += 1
            continue

        prefix = f"a{a_key}_"
        if C_key == ():
            counts[f"{prefix}empty_count"] += 1
        elif C_key == (0,):
            counts[f"{prefix}0_count"] += 1
        elif C_key == (1,):
            counts[f"{prefix}1_count"] += 1
        elif C_key == (0, 1):
            counts[f"{prefix}01_count"] += 1
        else:
            counts[f"{prefix}other_count"] += 1

    out: Dict[str, float] = {k: float(v) for k, v in counts.items()}
    denom = float(n_total) if n_total > 0 else float("nan")
    for k, v in counts.items():
        out[k.replace("_count", "_rate")] = float(v) / denom if n_total > 0 else float("nan")
    return out


def _validate_pair_counts(pair_stats: Dict[str, float], n_total: int, action1_count: int, method_name: str) -> None:
    other_total = int(
        round(
            pair_stats["a0_other_count"]
            + pair_stats["a1_other_count"]
            + pair_stats["other_action_count"]
        )
    )
    if other_total != 0:
        raise ValueError(f"{method_name} encountered unexpected action/set categories.")

    pair_count_total = 0
    for a in [0, 1]:
        for set_key in ["empty", "0", "1", "01"]:
            pair_count_total += int(round(pair_stats[f"a{a}_{set_key}_count"]))
    if pair_count_total != n_total:
        raise ValueError(f"{method_name} pair-count mismatch: total={pair_count_total}, n_test={n_total}.")

    pair_a1_total = 0
    for set_key in ["empty", "0", "1", "01"]:
        pair_a1_total += int(round(pair_stats[f"a1_{set_key}_count"]))
    if pair_a1_total != action1_count:
        raise ValueError(
            f"{method_name} action-count mismatch: pair_total={pair_a1_total}, action1_count={action1_count}."
        )


def _set01_rate_given_action0(pair_stats: Dict[str, float]) -> float:
    a0_total = sum(float(pair_stats[f"a0_{set_key}_count"]) for set_key in ["empty", "0", "1", "01"])
    if a0_total <= 0.0:
        return float("nan")
    return float(pair_stats["a0_01_count"]) / a0_total


def _row_cache_key(x: np.ndarray) -> str:
    x_arr = np.asarray(x, dtype=object).reshape(-1)
    parts: List[str] = []
    for v in x_arr.tolist():
        if isinstance(v, (float, np.floating)) and np.isnan(v):
            parts.append("__nan__")
        else:
            parts.append(repr(v))
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()


class CachedOutcomeModel:
    def __init__(self, base_model: OutcomeModel, action_set: List[Any]):
        self.base_model = base_model
        self.action_set = list(action_set)
        self._cache_by_action: Dict[Any, Dict[str, np.ndarray]] = {a: {} for a in self.action_set}

    def cache_split(self, X: np.ndarray) -> None:
        X = np.asarray(X, dtype=object)
        keys = [_row_cache_key(x) for x in X]
        for a in self.action_set:
            probs = np.asarray(self.base_model.predict_proba_batch(X, a), dtype=float)
            if probs.shape[0] != X.shape[0]:
                raise ValueError("Outcome cache build returned the wrong number of rows.")
            for k, p in zip(keys, probs):
                self._cache_by_action[a][k] = np.asarray(p, dtype=float)

    def predict_proba(self, x: np.ndarray, a: Any) -> np.ndarray:
        cached = self._cache_by_action.get(a, {}).get(_row_cache_key(x))
        if cached is not None:
            return np.asarray(cached, dtype=float).copy()
        return np.asarray(self.base_model.predict_proba(x, a), dtype=float)

    def predict_proba_batch(self, X: np.ndarray, a: Any) -> np.ndarray:
        X = np.asarray(X, dtype=object)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        out = np.zeros((X.shape[0], len(self.base_model.label_set)), dtype=float)
        missing_rows: List[int] = []
        missing_x: List[np.ndarray] = []
        a_cache = self._cache_by_action.get(a, {})
        for i, x in enumerate(X):
            cached = a_cache.get(_row_cache_key(x))
            if cached is None:
                missing_rows.append(i)
                missing_x.append(np.asarray(x, dtype=object))
            else:
                out[i] = np.asarray(cached, dtype=float)
        if missing_rows:
            probs = np.asarray(self.base_model.predict_proba_batch(np.asarray(missing_x, dtype=object), a), dtype=float)
            for local_i, row_i in enumerate(missing_rows):
                out[row_i] = probs[local_i]
                a_cache[_row_cache_key(X[row_i])] = np.asarray(probs[local_i], dtype=float)
        return out

    def predict_proba_all_actions(self, x: np.ndarray) -> Dict[Any, np.ndarray]:
        return {a: self.predict_proba(x, a) for a in self.action_set}


class CachedMarginalLabelModel:
    def __init__(self, base_model: MarginalLabelModel):
        self.base_model = base_model
        self._cache: Dict[str, np.ndarray] = {}

    def cache_split(self, X: np.ndarray) -> None:
        X = np.asarray(X, dtype=object)
        probs = np.asarray(self.base_model.predict_proba_batch(X), dtype=float)
        if probs.shape[0] != X.shape[0]:
            raise ValueError("Marginal cache build returned the wrong number of rows.")
        for x, p in zip(X, probs):
            self._cache[_row_cache_key(x)] = np.asarray(p, dtype=float)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        cached = self._cache.get(_row_cache_key(x))
        if cached is not None:
            return np.asarray(cached, dtype=float).copy()
        return np.asarray(self.base_model.predict_proba(x), dtype=float)

    def predict_proba_batch(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=object)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        out = np.zeros((X.shape[0], len(self.base_model.label_set)), dtype=float)
        missing_rows: List[int] = []
        missing_x: List[np.ndarray] = []
        for i, x in enumerate(X):
            cached = self._cache.get(_row_cache_key(x))
            if cached is None:
                missing_rows.append(i)
                missing_x.append(np.asarray(x, dtype=object))
            else:
                out[i] = np.asarray(cached, dtype=float)
        if missing_rows:
            probs = np.asarray(self.base_model.predict_proba_batch(np.asarray(missing_x, dtype=object)), dtype=float)
            for local_i, row_i in enumerate(missing_rows):
                out[row_i] = probs[local_i]
                self._cache[_row_cache_key(X[row_i])] = np.asarray(probs[local_i], dtype=float)
        return out


def randomized_test_coverage_estimate(
    sets: List[Any],
    actions: np.ndarray,
    A_test: np.ndarray,
    Y_test: np.ndarray,
) -> float:
    actions_arr = np.asarray(actions, dtype=int).reshape(-1)
    A_arr = np.asarray(A_test, dtype=int).reshape(-1)
    Y_arr = np.asarray(Y_test, dtype=int).reshape(-1)
    n = len(sets)

    if actions_arr.shape[0] != n or A_arr.shape[0] != n or Y_arr.shape[0] != n:
        raise ValueError("sets, actions, A_test, and Y_test must have the same length.")

    def is_covered(C: Any, y: int) -> bool:
        C_key = _canonicalize_prediction_set(C)
        return C_key is not None and len(C_key) > 0 and int(y) in C_key

    def subgroup_mean(mask: np.ndarray, expected_action: int) -> float:
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            return float("nan")
        hits = [
            1.0 if actions_arr[i] == expected_action and is_covered(sets[i], int(Y_arr[i])) else 0.0
            for i in idx
        ]
        return float(np.mean(hits))

    return float(subgroup_mean(A_arr == 1, 1) + subgroup_mean(A_arr == 0, 0))


def flatten_pair_stats(method_prefix: str, pair_stats: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for a in [0, 1]:
        for set_key in ["empty", "0", "1", "01"]:
            out[f"{method_prefix}_pair_a{a}_{set_key}_count"] = float(pair_stats[f"a{a}_{set_key}_count"])
            out[f"{method_prefix}_pair_a{a}_{set_key}_rate"] = float(pair_stats[f"a{a}_{set_key}_rate"])
    return out


def run_one_alpha(
    alpha: float,
    data: Dict[str, np.ndarray],
    action_set: List[Any],
    label_set: List[Any],
    outcome_model: OutcomeModel,
    behavior_model: BehaviorModel,
    marginal_model: MarginalLabelModel,
) -> Dict[str, float]:
    X_learn = data["X_learn"]
    X_cal, Y_cal, A_cal = data["X_cal"], data["Y_cal"], data["A_cal"]
    X_test, Y_test, A_test = data["X_test"], data["Y_test"], data["A_test"]

    pc_racp_cfg = PCRACPConformalPolicyConfig(
        alpha=alpha,
        action_set=action_set,
        label_set=label_set,
        utility_func=utility_func,
        u_max=U_MAX,
        beta_max=10.0,
        min_pi_prob=1e-6,
    )
    pc_racp = PCRACPConformalPolicy(pc_racp_cfg)
    pc_racp.learn_beta(X_learn, outcome_model)
    pc_sets, pc_actions, _ = pc_racp.build_prediction_sets(
        X_cal=X_cal,
        Y_cal=Y_cal,
        A_cal=A_cal,
        X_test=X_test,
        outcome_model=outcome_model,
        behavior_model=behavior_model,
    )

    pc_actions_arr = np.asarray(pc_actions, dtype=int)
    pc_utility = float(np.mean([fixed_action_worst_value(C, a, utility_func, U_MAX) for C, a in zip(pc_sets, pc_actions)]))
    pc_coverage = randomized_test_coverage_estimate(pc_sets, pc_actions_arr, A_test, Y_test)
    pc_action1_count = int(np.sum(pc_actions_arr == 1))
    pc_pair_stats = _count_action_set_pairs(pc_sets, pc_actions, int(X_test.shape[0]))
    _validate_pair_counts(pc_pair_stats, int(X_test.shape[0]), pc_action1_count, "PC-RACP")

    rac_cfg = RACConfig(
        alpha=alpha,
        action_set=action_set,
        label_set=label_set,
        utility_func=utility_func,
        u_max=U_MAX,
        beta_min=-10.0,
        beta_max=10.0,
        bisect_iters=60,
        bisect_tol=1e-7,
        eps=0.0,
    )
    rac = RiskAverseCalibration(rac_cfg)
    rac.fit(np.concatenate([data["X_learn"], X_cal], axis=0), np.concatenate([data["Y_learn"], Y_cal], axis=0), marginal_model)

    rac_sets: List[List[Any]] = []
    rac_actions: List[int] = []
    prob_test = marginal_model.predict_proba_batch(X_test)
    for i in range(X_test.shape[0]):
        C, _, action, _ = rac.predict_set_from_prob(prob_test[i])
        rac_sets.append(C)
        rac_actions.append(int(action))

    rac_actions_arr = np.asarray(rac_actions, dtype=int)
    rac_utility = float(np.mean([set_maxmin_value(C, utility_func, U_MAX, action_set) for C in rac_sets]))
    rac_coverage = randomized_test_coverage_estimate(rac_sets, rac_actions_arr, A_test, Y_test)
    rac_action1_count = int(np.sum(rac_actions_arr == 1))
    rac_pair_stats = _count_action_set_pairs(rac_sets, rac_actions, int(X_test.shape[0]))
    _validate_pair_counts(rac_pair_stats, int(X_test.shape[0]), rac_action1_count, "RAC")

    result: Dict[str, float] = {
        "alpha": float(alpha),
        "target_coverage": float(1.0 - alpha),
        "pc_racp_coverage": float(pc_coverage),
        "rac_coverage": float(rac_coverage),
        "pc_racp_utility": float(pc_utility),
        "rac_utility": float(rac_utility),
        "pc_racp_action1_count": float(pc_action1_count),
        "rac_action1_count": float(rac_action1_count),
        "pc_racp_action1_rate": float(np.mean(pc_actions_arr == 1)),
        "rac_action1_rate": float(np.mean(rac_actions_arr == 1)),
        "pc_racp_a0_set01_rate": float(_set01_rate_given_action0(pc_pair_stats)),
        "rac_a0_set01_rate": float(_set01_rate_given_action0(rac_pair_stats)),
        "n_test": float(X_test.shape[0]),
    }
    result.update(flatten_pair_stats("pc_racp", pc_pair_stats))
    result.update(flatten_pair_stats("rac", rac_pair_stats))
    return result


def print_summary_table(results: List[Dict[str, float]]) -> None:
    print()
    print("alpha | target | pc_racp_coverage | rac_coverage | pc_racp_utility | rac_utility")
    print("-" * 82)
    for r in results:
        print(
            f"{r['alpha']:.2f}  | "
            f"{r['target_coverage']:.2f}   | "
            f"{r['pc_racp_coverage']:.4f}           | "
            f"{r['rac_coverage']:.4f}       | "
            f"{r['pc_racp_utility']:.4f}          | "
            f"{r['rac_utility']:.4f}"
        )


def save_plots(results: List[Dict[str, float]], out_dir: str) -> None:
    import matplotlib.pyplot as plt

    set_paper_style(plt)
    os.makedirs(out_dir, exist_ok=True)

    alphas = [r["alpha"] for r in results]
    target_coverage = [r["target_coverage"] for r in results]

    fig, ax = plt.subplots(figsize=PLOT_STYLE["figsize"])
    ax.plot(alphas, [r["pc_racp_utility"] for r in results], marker="o", linewidth=PLOT_STYLE["line_width"], markersize=PLOT_STYLE["marker_size"], color="#d62728", label="PC-RACP")
    ax.plot(alphas, [r["rac_utility"] for r in results], marker="o", linewidth=PLOT_STYLE["line_width"], markersize=PLOT_STYLE["marker_size"], color="#1f77b4", label="RAC")
    style_axis(ax, r"Utility vs. $\alpha$", "Utility")
    add_legend(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, PLOT_FILENAMES["utility_pdf"]), dpi=PLOT_STYLE["dpi"], bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, PLOT_FILENAMES["utility_png"]), dpi=PLOT_STYLE["dpi"], bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=PLOT_STYLE["figsize"])
    ax.plot(alphas, [r["pc_racp_coverage"] for r in results], marker="o", linewidth=PLOT_STYLE["line_width"], markersize=PLOT_STYLE["marker_size"], color="#d62728", label="PC-RACP")
    ax.plot(alphas, [r["rac_coverage"] for r in results], marker="o", linewidth=PLOT_STYLE["line_width"], markersize=PLOT_STYLE["marker_size"], color="#1f77b4", label="RAC")
    ax.plot(alphas, target_coverage, linestyle="--", linewidth=PLOT_STYLE["line_width"], color="#4d4d4d", label=r"Target $1-\alpha$")
    coverage_all = np.asarray(
        [r["pc_racp_coverage"] for r in results] + [r["rac_coverage"] for r in results] + target_coverage,
        dtype=float,
    )
    coverage_all = coverage_all[np.isfinite(coverage_all)]
    if coverage_all.size:
        y_min = float(np.min(coverage_all))
        y_max = float(np.max(coverage_all))
        y_span = max(y_max - y_min, 1e-3)
        y_pad = max(0.01, 0.12 * y_span)
        ax.set_ylim(max(0.0, y_min - y_pad), min(1.0, y_max + y_pad))
    style_axis(ax, r"Coverage vs. $\alpha$", "Coverage")
    add_legend(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, PLOT_FILENAMES["coverage_pdf"]), dpi=PLOT_STYLE["dpi"], bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, PLOT_FILENAMES["coverage_png"]), dpi=PLOT_STYLE["dpi"], bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=PLOT_STYLE["figsize"])
    ax.plot(alphas, [r["pc_racp_action1_count"] for r in results], marker="o", linewidth=PLOT_STYLE["line_width"], markersize=PLOT_STYLE["marker_size"], color="#d62728", label="PC-RACP")
    ax.plot(alphas, [r["rac_action1_count"] for r in results], marker="o", linewidth=PLOT_STYLE["line_width"], markersize=PLOT_STYLE["marker_size"], color="#1f77b4", label="RAC")
    style_axis(ax, r"Selected action $A=1$ count vs. $\alpha$", "Count on test")
    add_legend(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, PLOT_FILENAMES["action1_count_pdf"]), dpi=PLOT_STYLE["dpi"], bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, PLOT_FILENAMES["action1_count_png"]), dpi=PLOT_STYLE["dpi"], bbox_inches="tight")
    plt.close(fig)

    pair_curve_specs = [
        ("empty", "empty", "#4d4d4d"),
        ("0", "{0}", "#1f77b4"),
        ("1", "{1}", "#ff7f0e"),
        ("01", "{0,1}", "#2ca02c"),
    ]

    for action_value in [0, 1]:
        fig, ax = plt.subplots(figsize=PLOT_STYLE["wide_figsize"])
        for method_prefix, method_label, line_style, marker_style in [
            ("pc_racp", "PC-RACP", "-", "o"),
            ("rac", "RAC", "--", "s"),
        ]:
            for set_key, label, color in pair_curve_specs:
                values = [r[f"{method_prefix}_pair_a{action_value}_{set_key}_count"] for r in results]
                ax.plot(
                    alphas,
                    values,
                    linestyle=line_style,
                    marker=marker_style,
                    linewidth=PLOT_STYLE["line_width"],
                    markersize=PLOT_STYLE["marker_size"],
                    color=color,
                    label=f"{method_label} {label}",
                )
        style_axis(ax, f"Prediction-set counts given action a={action_value}", "Count on test")
        add_legend(ax, ncol=2)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, PLOT_FILENAMES[f"pair_counts_a{action_value}_pdf"]), dpi=PLOT_STYLE["dpi"], bbox_inches="tight")
        fig.savefig(os.path.join(out_dir, PLOT_FILENAMES[f"pair_counts_a{action_value}_png"]), dpi=PLOT_STYLE["dpi"], bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=PLOT_STYLE["figsize"])
    ax.plot(alphas, [r["pc_racp_a0_set01_rate"] for r in results], marker="o", linewidth=PLOT_STYLE["line_width"], markersize=PLOT_STYLE["marker_size"], color="#d62728", label="PC-RACP")
    ax.plot(alphas, [r["rac_a0_set01_rate"] for r in results], marker="o", linewidth=PLOT_STYLE["line_width"], markersize=PLOT_STYLE["marker_size"], color="#1f77b4", label="RAC")
    rate_all = np.asarray([r["pc_racp_a0_set01_rate"] for r in results] + [r["rac_a0_set01_rate"] for r in results], dtype=float)
    rate_all = rate_all[np.isfinite(rate_all)]
    if rate_all.size:
        y_min = float(np.min(rate_all))
        y_max = float(np.max(rate_all))
        y_span = max(y_max - y_min, 1e-3)
        y_pad = max(0.01, 0.12 * y_span)
        ax.set_ylim(max(0.0, y_min - y_pad), min(1.0, y_max + y_pad))
    style_axis(ax, r"Rate of $C=\{0,1\}$ among selected $A=0$ vs. $\alpha$", "Rate")
    add_legend(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, PLOT_FILENAMES["a0_set01_rate_pdf"]), dpi=PLOT_STYLE["dpi"], bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, PLOT_FILENAMES["a0_set01_rate_png"]), dpi=PLOT_STYLE["dpi"], bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    alphas = parse_alpha_grid(args.alphas)
    catboost_params = catboost_params_from_args(args)

    X, Y, A, meta = generate_data(
        n_samples=args.n_samples,
        random_state=args.seed,
        return_env=True,
        covariate_cols=DEFAULT_COVARIATE_COLS,
        standardize=False,
    )

    action_set = sorted(set(A.tolist()))
    label_set = sorted(set(Y.tolist()))
    covariates = list(meta.get("covariate_cols_used", DEFAULT_COVARIATE_COLS))
    categorical_features = list(meta.get("categorical_feature_names", []))

    data = split_data(X, Y, A, train_frac=0.3, learn_frac=0.2, calib_frac=0.2, random_state=args.seed)
    print("Hillstrom experiment")
    print(f"n={X.shape[0]}, seed={args.seed}, alphas={alphas}")
    print(
        "split sizes: "
        f"train={data['X_train'].shape[0]}, learn={data['X_learn'].shape[0]}, "
        f"calib={data['X_cal'].shape[0]}, test={data['X_test'].shape[0]}"
    )
    print(f"covariates: {covariates}")

    outcome_model_raw = OutcomeModel(
        action_set=action_set,
        label_set=label_set,
        feature_names=covariates,
        categorical_feature_names=categorical_features,
        catboost_params=catboost_params,
    )
    outcome_model_raw.fit(data["X_train"], data["A_train"], data["Y_train"])

    behavior_model = BehaviorModel(action_set=action_set)
    behavior_model.fit(data["X_train"], data["A_train"])

    marginal_model_raw = MarginalLabelModel(
        label_set=label_set,
        feature_names=covariates,
        categorical_feature_names=categorical_features,
        catboost_params=catboost_params,
    )
    marginal_model_raw.fit(data["X_train"], data["Y_train"])

    outcome_model = CachedOutcomeModel(outcome_model_raw, action_set=action_set)
    marginal_model = CachedMarginalLabelModel(marginal_model_raw)
    for split_name in ["X_learn", "X_cal", "X_test"]:
        outcome_model.cache_split(data[split_name])
        marginal_model.cache_split(data[split_name])

    results: List[Dict[str, float]] = []
    for alpha in alphas:
        result = run_one_alpha(alpha, data, action_set, label_set, outcome_model, behavior_model, marginal_model)
        results.append(result)
        print(
            f"alpha={alpha:.2f} | "
            f"pc_racp_coverage={result['pc_racp_coverage']:.4f}, "
            f"rac_coverage={result['rac_coverage']:.4f}, "
            f"pc_racp_utility={result['pc_racp_utility']:.4f}, "
            f"rac_utility={result['rac_utility']:.4f}"
        )

    print_summary_table(results)

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "alpha_sweep.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    save_plots(results, args.out_dir)
    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved plots under: {args.out_dir}")


if __name__ == "__main__":
    main()
