from __future__ import annotations

import argparse
import contextlib
import csv
import io
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

THREAD_LIMIT_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
for _env_name in THREAD_LIMIT_ENV_VARS:
    os.environ[_env_name] = "1"
os.environ.setdefault("PYTHONWARNINGS", "ignore")

import numpy as np

from src.data_generator import LinearSoftmaxEnvironment, generate_data, utility_func, U_MAX
from src.model import (
    BehaviorModel,
    MarginalLabelModel,
    OutcomeModel,
    SUPPORTED_MODEL_TYPES,
)
from src.method_pc_racp import PCRACPConformalPolicy, PCRACPConformalPolicyConfig
from src.method_rac import RACConfig, RiskAverseCalibration
from src.method_direct_plugin import DirectPluginPolicy, DirectPluginPolicyConfig


PLOT_STYLE = {
    "figsize": (3.4, 2.5),
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
    "utility_png": "utility_vs_alpha_logistic.png",
    "utility_pdf": "utility_vs_alpha_logistic.pdf",
    "coverage_png": "coverage_vs_alpha_logistic.png",
    "coverage_pdf": "coverage_vs_alpha_logistic.pdf",
}

PLOT_MODEL_SUFFIXES = {
    "logistic": "logistic",
    "random_forest": "rf",
}

DEFAULT_N_SAMPLES = 30000
DEFAULT_ALPHAS = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]


def configure_runtime_warnings():
    """
    Disable runtime warnings to keep experiment output clean.
    """
    # Process-local suppression.
    warnings.simplefilter("ignore")
    warnings.filterwarnings("ignore")

    # Propagate to child processes / joblib workers.
    os.environ["PYTHONWARNINGS"] = "ignore"


def set_paper_style(plt):
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


def style_axis(ax, title: str, y_label: str):
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
    ax.legend(
        loc="best",
        frameon=True,
        framealpha=0.85,
        edgecolor=PLOT_STYLE["legend_edge_color"],
        facecolor="white",
    )


def plot_filenames(model_type: str) -> Dict[str, str]:
    filenames = dict(PLOT_FILENAMES)
    suffix = PLOT_MODEL_SUFFIXES[str(model_type)]
    filenames["utility_png"] = f"utility_vs_alpha_{suffix}.png"
    filenames["utility_pdf"] = f"utility_vs_alpha_{suffix}.pdf"
    filenames["coverage_png"] = f"coverage_vs_alpha_{suffix}.png"
    filenames["coverage_pdf"] = f"coverage_vs_alpha_{suffix}.pdf"
    return filenames


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
        raise ValueError("calib_frac is too small; need at least 1 calibration sample.")

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
    """PC-RACP algorithm utility: min_{y in C(x)} u(a_hat(x), y), empty set -> u_max."""
    if C_x is None or len(C_x) == 0:
        return float(u_max)
    return float(min(float(u_func(a, y)) for y in C_x))


def set_maxmin_value(C, utility_func, u_max: float, action_set):
    """RAC utility: v(C) = max_a min_{y in C} u(a,y). Empty set -> u_max."""
    if C is None or len(C) == 0:
        return float(u_max)
    worsts = []
    for a in action_set:
        worsts.append(min(float(utility_func(a, y)) for y in C))
    return float(max(worsts))


def sample_labels_from_actions_true_env(
    X: np.ndarray,
    actions: np.ndarray,
    true_env: LinearSoftmaxEnvironment,
    rng: np.random.Generator,
) -> np.ndarray:
    X = np.asarray(X)
    actions = np.asarray(actions)
    if X.shape[0] != actions.shape[0]:
        raise ValueError("X and actions must have the same number of rows.")

    y = np.empty(X.shape[0], dtype=int)
    for idx, (x, a) in enumerate(zip(X, actions)):
        p = np.asarray(true_env.outcome_proba(x, int(a)), dtype=float)
        if p.ndim != 1 or p.size == 0:
            raise ValueError("true_env.outcome_proba must return a non-empty 1D probability vector.")
        if not np.all(np.isfinite(p)):
            raise ValueError("true_env.outcome_proba returned non-finite values.")
        if np.any(p < 0.0):
            raise ValueError("true_env.outcome_proba returned negative probabilities.")

        s = float(np.sum(p))
        if not np.isclose(s, 1.0):
            if s <= 0.0:
                p = np.ones_like(p, dtype=float) / float(p.size)
            else:
                p = p / s
        y[idx] = int(rng.choice(p.size, p=p))
    return y.astype(int)


def _run_one_alpha_with_reusable_methods(
    alpha: float,
    data: Dict[str, np.ndarray],
    action_set: List[Any],
    outcome_model: OutcomeModel,
    plugin_outcome_model: OutcomeModel,
    behavior_model: BehaviorModel,
    true_env: LinearSoftmaxEnvironment,
    off: PCRACPConformalPolicy,
    direct: DirectPluginPolicy,
    rac: RiskAverseCalibration,
    eval_seed: int,
    pi_cal_matrix: np.ndarray | None = None,
    pi_test_matrix: np.ndarray | None = None,
    rac_prob_test: np.ndarray | None = None,
) -> Dict[str, float]:
    X_learn = data["X_learn"]
    X_cal, Y_cal, A_cal = data["X_cal"], data["Y_cal"], data["A_cal"]
    X_test = data["X_test"]

    off.set_alpha(alpha)
    off.learn_beta(X_learn, outcome_model)

    eval_seed_seq = np.random.default_rng(int(eval_seed))
    pc_seed = int(eval_seed_seq.integers(0, np.iinfo(np.uint32).max))
    plugin_seed = int(eval_seed_seq.integers(0, np.iinfo(np.uint32).max))
    rac_seed = int(eval_seed_seq.integers(0, np.iinfo(np.uint32).max))

    sets, acts, _ = off.build_prediction_sets(
        X_cal=X_cal,
        Y_cal=Y_cal,
        A_cal=A_cal,
        X_test=X_test,
        outcome_model=outcome_model,
        behavior_model=behavior_model,
        pi_cal_matrix=pi_cal_matrix,
        pi_test_matrix=pi_test_matrix,
    )

    fixed_vals = [fixed_action_worst_value(C, a, utility_func, U_MAX) for C, a in zip(sets, acts)]
    fixed_avg = float(np.mean(fixed_vals)) if len(fixed_vals) else float("nan")

    rng = np.random.default_rng(int(pc_seed))
    Y_test_policy = sample_labels_from_actions_true_env(X_test, acts, true_env, rng)
    hits_policy = []
    for y_policy, C in zip(Y_test_policy, sets):
        if C is None or len(C) == 0:
            hits_policy.append(0.0)
        else:
            hits_policy.append(1.0 if int(y_policy) in C else 0.0)
    cov_emp_policyY = float(np.mean(hits_policy)) if len(hits_policy) else float("nan")

    direct.set_alpha(alpha)
    sets_direct, acts_direct, _ = direct.build_prediction_sets(
        X_test=X_test,
        outcome_model=plugin_outcome_model,
    )

    direct_vals = [
        fixed_action_worst_value(C, a, utility_func, U_MAX) for C, a in zip(sets_direct, acts_direct)
    ]
    direct_avg_u = float(np.mean(direct_vals)) if len(direct_vals) else float("nan")

    rng = np.random.default_rng(plugin_seed)
    Y_test_direct_policy = sample_labels_from_actions_true_env(X_test, acts_direct, true_env, rng)
    hits_direct_policy = []
    for y_policy, C in zip(Y_test_direct_policy, sets_direct):
        if C is None or len(C) == 0:
            hits_direct_policy.append(0.0)
        else:
            hits_direct_policy.append(1.0 if int(y_policy) in C else 0.0)
    direct_cov_emp_policyY = float(np.mean(hits_direct_policy)) if len(hits_direct_policy) else float("nan")

    rac.set_alpha(alpha)
    if rac_prob_test is not None and len(rac_prob_test) != len(X_test):
        raise ValueError("rac_prob_test must have the same number of rows as X_test.")

    sets_rac = []
    acts_rac = []
    for idx, x in enumerate(X_test):
        if rac_prob_test is None:
            C, _, a_star, _ = rac.predict_set(x.reshape(1, -1))
        else:
            C, _, a_star, _ = rac.predict_set_from_prob(rac_prob_test[idx])
        sets_rac.append(C)
        acts_rac.append(a_star)

    rac_vals = [set_maxmin_value(C, utility_func, U_MAX, action_set) for C in sets_rac]
    rac_avg_u = float(np.mean(rac_vals)) if len(rac_vals) else float("nan")

    rng = np.random.default_rng(rac_seed)
    Y_test_rac_policy = sample_labels_from_actions_true_env(X_test, acts_rac, true_env, rng)
    hits_rac = []
    for y_policy, C in zip(Y_test_rac_policy, sets_rac):
        if C is None or len(C) == 0:
            hits_rac.append(0.0)
        else:
            hits_rac.append(1.0 if int(y_policy) in C else 0.0)
    rac_cov_emp_policyY = float(np.mean(hits_rac)) if len(hits_rac) else float("nan")

    return {
        "alpha": float(alpha),
        "target_coverage": float(1.0 - alpha),
        "pc_racp_coverage": cov_emp_policyY,
        "rac_coverage": float(rac_cov_emp_policyY),
        "plugin_coverage": float(direct_cov_emp_policyY),
        "pc_racp_utility": fixed_avg,
        "rac_utility": float(rac_avg_u),
        "plugin_utility": float(direct_avg_u),
    }


def run_one_alpha(
    alpha: float,
    data: Dict[str, np.ndarray],
    action_set: List[Any],
    label_set: List[Any],
    outcome_model: OutcomeModel,
    plugin_outcome_model: OutcomeModel,
    behavior_model: BehaviorModel,
    marginal_model: MarginalLabelModel,
    true_env: LinearSoftmaxEnvironment,
    eval_seed: int,
) -> Dict[str, float]:
    outcome_probs_learn = outcome_model.predict_proba_all_actions_batch(data["X_learn"])
    outcome_probs_cal = outcome_model.predict_proba_all_actions_batch(data["X_cal"])
    outcome_probs_test = outcome_model.predict_proba_all_actions_batch(data["X_test"])
    plugin_probs_test = plugin_outcome_model.predict_proba_all_actions_batch(data["X_test"])

    off_cfg = PCRACPConformalPolicyConfig(
        alpha=alpha,
        action_set=action_set,
        label_set=label_set,
        utility_func=utility_func,
        u_max=U_MAX,
        beta_max=10.0,
        min_pi_prob=1e-6,
    )
    off = PCRACPConformalPolicy(off_cfg)
    off.precompute_candidates(
        X=data["X_learn"],
        outcome_model=outcome_model,
        probs_by_action_batch=outcome_probs_learn,
    )
    off.precompute_candidates(
        X=data["X_cal"],
        outcome_model=outcome_model,
        probs_by_action_batch=outcome_probs_cal,
    )
    off.precompute_candidates(
        X=data["X_test"],
        outcome_model=outcome_model,
        probs_by_action_batch=outcome_probs_test,
    )

    direct_cfg = DirectPluginPolicyConfig(
        alpha=alpha,
        action_set=action_set,
        label_set=label_set,
        utility_func=utility_func,
        u_max=U_MAX,
        tie_tol=1e-12,
    )
    direct = DirectPluginPolicy(direct_cfg)
    direct.precompute_candidates(
        X=data["X_test"],
        outcome_model=plugin_outcome_model,
        probs_by_action_batch=plugin_probs_test,
    )

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
    X_learn, Y_learn = data["X_learn"], data["Y_learn"]
    X_cal, Y_cal = data["X_cal"], data["Y_cal"]
    X_rac_cal = np.concatenate([X_learn, X_cal], axis=0)
    Y_rac_cal = np.concatenate([Y_learn, Y_cal], axis=0)
    rac.fit(X_rac_cal, Y_rac_cal, marginal_model)

    pi_cal_matrix = None
    pi_test_matrix = None
    if hasattr(behavior_model, "predict_action_proba_batch"):
        pi_cal_matrix = np.asarray(behavior_model.predict_action_proba_batch(data["X_cal"]), dtype=float)
        pi_test_matrix = np.asarray(behavior_model.predict_action_proba_batch(data["X_test"]), dtype=float)

    rac_prob_test = None
    if hasattr(marginal_model, "predict_proba_batch"):
        rac_prob_test = np.asarray(marginal_model.predict_proba_batch(data["X_test"]), dtype=float)

    return _run_one_alpha_with_reusable_methods(
        alpha=alpha,
        data=data,
        action_set=action_set,
        outcome_model=outcome_model,
        plugin_outcome_model=plugin_outcome_model,
        behavior_model=behavior_model,
        true_env=true_env,
        off=off,
        direct=direct,
        rac=rac,
        eval_seed=eval_seed,
        pi_cal_matrix=pi_cal_matrix,
        pi_test_matrix=pi_test_matrix,
        rac_prob_test=rac_prob_test,
    )


def print_two_tables(results: List[Dict[str, float]]):
    print()
    print("===== Table 1: alpha vs utilities =====")
    header1 = " alpha | pc_racp_utility | rac_utility | plugin_utility"
    print(header1)
    print("-" * len(header1))
    for r in results:
        print(
            f" {r['alpha']:.2f}  | "
            f"{r['pc_racp_utility']:.4f}             | "
            f"{r['rac_utility']:.4f}       | "
            f"{r['plugin_utility']:.4f}"
        )

    print()
    print("===== Table 2: alpha vs coverage =====")
    header2 = " alpha | target_coverage | pc_racp_coverage | rac_coverage | plugin_coverage"
    print(header2)
    print("-" * len(header2))
    for r in results:
        print(
            f" {r['alpha']:.2f}  | "
            f"{r['target_coverage']:.4f}          | "
            f"{r['pc_racp_coverage']:.4f}           | "
            f"{r['rac_coverage']:.4f}       | "
            f"{r['plugin_coverage']:.4f}"
        )

def save_plots(results: List[Dict[str, float]], out_dir: str, model_type: str = "logistic"):
    import matplotlib.pyplot as plt

    filenames = plot_filenames(model_type)
    set_paper_style(plt)

    alphas = [r["alpha"] for r in results]
    pc_racp_utility = [r["pc_racp_utility"] for r in results]
    rac_utility = [r["rac_utility"] for r in results]
    plugin_utility = [r["plugin_utility"] for r in results]
    pc_racp_coverage = [r["pc_racp_coverage"] for r in results]
    rac_coverage = [r["rac_coverage"] for r in results]
    plugin_coverage = [r["plugin_coverage"] for r in results]
    target_coverage = [r["target_coverage"] for r in results]

    fig, ax = plt.subplots(figsize=PLOT_STYLE["figsize"])
    ax.plot(
        alphas,
        pc_racp_utility,
        marker="o",
        linewidth=PLOT_STYLE["line_width"],
        markersize=PLOT_STYLE["marker_size"],
        color="#d62728",
        label="PC-RACP",
    )
    ax.plot(
        alphas,
        rac_utility,
        marker="o",
        linewidth=PLOT_STYLE["line_width"],
        markersize=PLOT_STYLE["marker_size"],
        color="#1f77b4",
        label="RAC",
    )
    ax.plot(
        alphas,
        plugin_utility,
        marker="o",
        linewidth=PLOT_STYLE["line_width"],
        markersize=PLOT_STYLE["marker_size"],
        color="#2ca02c",
        label="Plug-in",
    )
    style_axis(ax, r"Utility vs. $\alpha$", "Utility")
    fig.tight_layout()
    util_pdf_path = os.path.join(out_dir, filenames["utility_pdf"])
    util_png_path = os.path.join(out_dir, filenames["utility_png"])
    fig.savefig(util_pdf_path, dpi=PLOT_STYLE["dpi"], bbox_inches="tight")
    fig.savefig(util_png_path, dpi=PLOT_STYLE["dpi"], bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=PLOT_STYLE["figsize"])
    ax.plot(
        alphas,
        pc_racp_coverage,
        marker="o",
        linewidth=PLOT_STYLE["line_width"],
        markersize=PLOT_STYLE["marker_size"],
        color="#d62728",
        label="PC-RACP",
    )
    ax.plot(
        alphas,
        rac_coverage,
        marker="o",
        linewidth=PLOT_STYLE["line_width"],
        markersize=PLOT_STYLE["marker_size"],
        color="#1f77b4",
        label="RAC",
    )
    ax.plot(
        alphas,
        plugin_coverage,
        marker="o",
        linewidth=PLOT_STYLE["line_width"],
        markersize=PLOT_STYLE["marker_size"],
        color="#2ca02c",
        label="Plug-in",
    )
    ax.plot(
        alphas,
        target_coverage,
        linestyle="--",
        linewidth=PLOT_STYLE["line_width"],
        color="#4d4d4d",
        label=r"Target $1-\alpha$",
    )
    coverage_all = np.asarray(
        pc_racp_coverage + rac_coverage + plugin_coverage + target_coverage, dtype=float
    )
    y_min = float(np.min(coverage_all))
    y_max = float(np.max(coverage_all))
    y_span = max(y_max - y_min, 1e-3)
    y_pad = max(0.01, 0.12 * y_span)
    ax.set_ylim(max(0.0, y_min - y_pad), min(1.0, y_max + y_pad))
    style_axis(ax, r"Coverage vs. $\alpha$", "Coverage")
    fig.tight_layout()
    cov_pdf_path = os.path.join(out_dir, filenames["coverage_pdf"])
    cov_png_path = os.path.join(out_dir, filenames["coverage_png"])
    fig.savefig(cov_pdf_path, dpi=PLOT_STYLE["dpi"], bbox_inches="tight")
    fig.savefig(cov_png_path, dpi=PLOT_STYLE["dpi"], bbox_inches="tight")
    plt.close(fig)

    print()
    print("Saved plots to:")
    print(" ", util_pdf_path)
    print(" ", util_png_path)
    print(" ", cov_pdf_path)
    print(" ", cov_png_path)


def default_out_dir() -> str:
    project_dir = Path(__file__).resolve().parent
    return str(project_dir / "results" / "simulation")


def build_model_out_dir(base_out_dir: str, model_type: str) -> str:
    return os.path.join(os.path.abspath(base_out_dir), str(model_type))


def seed_csv_name(seed: int) -> str:
    return f"simulation_seed_{int(seed):02d}.csv"


def run_experiment(
    seed: int,
    n_samples: int,
    alphas: List[float] | None = None,
    model_type: str = "logistic",
) -> List[Dict[str, float]]:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if alphas is None:
        alphas = list(DEFAULT_ALPHAS)

    # Keep global NumPy RNG deterministic for any downstream code paths that use it.
    np.random.seed(int(seed))
    alpha_eval_seed_rng = np.random.default_rng(int(seed))

    print("PC-RACP / RAC / Plug-in comparison enabled.")
    print("Policy-induced coverage uses counterfactual outcomes sampled from the true simulation environment.")
    print(f"seed={int(seed)}")
    print(f"model_type={model_type}")

    X, Y, A, true_env = generate_data(n_samples=n_samples, random_state=int(seed), return_env=True)

    action_set = sorted(list(set(A.tolist())))
    label_set = sorted(list(set(Y.tolist())))
    print(f"action_set={action_set}, label_set={label_set}")

    data = split_data(X, Y, A, train_frac=0.3, learn_frac=0.2, calib_frac=0.2, random_state=int(seed))
    X_train, Y_train, A_train = data["X_train"], data["Y_train"], data["A_train"]
    print(
        "train:",
        X_train.shape[0],
        "learn:",
        data["X_learn"].shape[0],
        "calib:",
        data["X_cal"].shape[0],
        "test:",
        data["X_test"].shape[0],
    )

    outcome_model = OutcomeModel(
        action_set=action_set,
        label_set=label_set,
        model_type=model_type,
        random_state=int(seed),
    )
    outcome_model.fit(X_train, A_train, Y_train)

    X_plugin_train = np.concatenate([data["X_train"], data["X_learn"], data["X_cal"]], axis=0)
    A_plugin_train = np.concatenate([data["A_train"], data["A_learn"], data["A_cal"]], axis=0)
    Y_plugin_train = np.concatenate([data["Y_train"], data["Y_learn"], data["Y_cal"]], axis=0)
    plugin_outcome_model = OutcomeModel(
        action_set=action_set,
        label_set=label_set,
        model_type=model_type,
        random_state=int(seed),
    )
    plugin_outcome_model.fit(X_plugin_train, A_plugin_train, Y_plugin_train)

    outcome_probs_learn = outcome_model.predict_proba_all_actions_batch(data["X_learn"])
    outcome_probs_cal = outcome_model.predict_proba_all_actions_batch(data["X_cal"])
    outcome_probs_test = outcome_model.predict_proba_all_actions_batch(data["X_test"])
    plugin_probs_test = plugin_outcome_model.predict_proba_all_actions_batch(data["X_test"])

    behavior_model = BehaviorModel(
        action_set=action_set,
        model_type=model_type,
        random_state=int(seed),
    )
    behavior_model.fit(X_train, A_train)

    marginal_model = MarginalLabelModel(
        label_set=label_set,
        model_type=model_type,
        random_state=int(seed),
    )
    marginal_model.fit(X_train, Y_train)

    alpha_init = float(alphas[0]) if len(alphas) > 0 else float(DEFAULT_ALPHAS[0])

    off_cfg = PCRACPConformalPolicyConfig(
        alpha=alpha_init,
        action_set=action_set,
        label_set=label_set,
        utility_func=utility_func,
        u_max=U_MAX,
        beta_max=10.0,
        min_pi_prob=1e-6,
    )
    off = PCRACPConformalPolicy(off_cfg)
    off.precompute_candidates(
        X=data["X_learn"],
        outcome_model=outcome_model,
        probs_by_action_batch=outcome_probs_learn,
    )
    off.precompute_candidates(
        X=data["X_cal"],
        outcome_model=outcome_model,
        probs_by_action_batch=outcome_probs_cal,
    )
    off.precompute_candidates(
        X=data["X_test"],
        outcome_model=outcome_model,
        probs_by_action_batch=outcome_probs_test,
    )

    direct_cfg = DirectPluginPolicyConfig(
        alpha=alpha_init,
        action_set=action_set,
        label_set=label_set,
        utility_func=utility_func,
        u_max=U_MAX,
        tie_tol=1e-12,
    )
    direct = DirectPluginPolicy(direct_cfg)
    direct.precompute_candidates(
        X=data["X_test"],
        outcome_model=plugin_outcome_model,
        probs_by_action_batch=plugin_probs_test,
    )

    rac_cfg = RACConfig(
        alpha=alpha_init,
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
    X_rac_cal = np.concatenate([data["X_learn"], data["X_cal"]], axis=0)
    Y_rac_cal = np.concatenate([data["Y_learn"], data["Y_cal"]], axis=0)
    rac.fit(X_rac_cal, Y_rac_cal, marginal_model)

    pi_cal_matrix = None
    pi_test_matrix = None
    if hasattr(behavior_model, "predict_action_proba_batch"):
        pi_cal_matrix = np.asarray(
            behavior_model.predict_action_proba_batch(data["X_cal"]),
            dtype=float,
        )
        pi_test_matrix = np.asarray(
            behavior_model.predict_action_proba_batch(data["X_test"]),
            dtype=float,
        )

    rac_prob_test = None
    if hasattr(marginal_model, "predict_proba_batch"):
        rac_prob_test = np.asarray(
            marginal_model.predict_proba_batch(data["X_test"]),
            dtype=float,
        )

    results: List[Dict[str, float]] = []

    for alpha in alphas:
        eval_seed = int(alpha_eval_seed_rng.integers(0, np.iinfo(np.uint32).max))
        r = _run_one_alpha_with_reusable_methods(
            alpha=alpha,
            data=data,
            action_set=action_set,
            outcome_model=outcome_model,
            plugin_outcome_model=plugin_outcome_model,
            behavior_model=behavior_model,
            true_env=true_env,
            off=off,
            direct=direct,
            rac=rac,
            eval_seed=eval_seed,
            pi_cal_matrix=pi_cal_matrix,
            pi_test_matrix=pi_test_matrix,
            rac_prob_test=rac_prob_test,
        )
        r["seed"] = int(seed)
        results.append(r)

        print(
            f"alpha={r['alpha']:.2f} | "
            f"PC-RACP utility={r['pc_racp_utility']:.4f}, "
            f"coverage={r['pc_racp_coverage']:.4f} | "
            f"RAC utility={r['rac_utility']:.4f}, coverage={r['rac_coverage']:.4f} | "
            f"Plug-in utility={r['plugin_utility']:.4f}, "
            f"coverage={r['plugin_coverage']:.4f}"
        )

    return results


def save_results_csv(results: List[Dict[str, float]], csv_path: str):
    if not results:
        raise ValueError("No results to write.")
    destination = Path(csv_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    temporary.replace(destination)
    print(f"\nSaved CSV: {csv_path}")


UINT32_MAX = int(np.iinfo(np.uint32).max)
MIN_NUM_SEEDS = 2
PER_SEED_COLUMNS = (
    "alpha",
    "target_coverage",
    "pc_racp_coverage",
    "rac_coverage",
    "plugin_coverage",
    "pc_racp_utility",
    "rac_utility",
    "plugin_utility",
    "seed",
)
SUMMARY_METRICS = (
    "target_coverage",
    "pc_racp_coverage",
    "rac_coverage",
    "plugin_coverage",
    "pc_racp_utility",
    "rac_utility",
    "plugin_utility",
)

SUMMARY_CSV_COLUMNS = ("alpha", *SUMMARY_METRICS)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the paper simulation over multiple random seeds and produce "
            "per-seed, combined, and aggregated results."
        )
    )
    parser.add_argument(
        "--start_seed",
        type=int,
        default=0,
        help="First seed in the consecutive seed range (default: 0).",
    )
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=20,
        help="Number of consecutive seeds; must be at least 2 (default: 20).",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=5,
        help="Maximum number of seed processes to run concurrently (default: 5).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=default_out_dir(),
        help="Base output directory (default: results/simulation).",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=DEFAULT_N_SAMPLES,
        help=f"Synthetic sample size for every seed (default: {DEFAULT_N_SAMPLES}).",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=list(SUPPORTED_MODEL_TYPES),
        default="logistic",
        help="Estimator backend used by all methods.",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> List[int]:
    if int(args.num_seeds) < MIN_NUM_SEEDS:
        raise ValueError(f"num_seeds must be at least {MIN_NUM_SEEDS}.")
    if int(args.max_workers) <= 0:
        raise ValueError("max_workers must be positive.")
    if int(args.n_samples) <= 0:
        raise ValueError("n_samples must be positive.")
    if int(args.start_seed) < 0:
        raise ValueError("start_seed must be non-negative.")

    last_seed = int(args.start_seed) + int(args.num_seeds) - 1
    if last_seed > UINT32_MAX:
        raise ValueError(f"Seeds must not exceed {UINT32_MAX}.")
    return list(range(int(args.start_seed), last_seed + 1))


def validate_output_directory(
    out_dir: Path,
    seeds: Sequence[int],
) -> None:
    if not out_dir.exists():
        return

    expected_seed_files = {seed_csv_name(seed) for seed in seeds}
    existing_seed_files = {path.name for path in out_dir.glob("simulation_seed_*.csv")}
    if existing_seed_files and existing_seed_files != expected_seed_files:
        raise RuntimeError(
            f"{out_dir} contains a different set of per-seed CSV files. "
            "Choose a different --out_dir to avoid mixing experiments."
        )


def _run_seed(seed: int, out_dir: str, n_samples: int, model_type: str) -> str:
    configure_runtime_warnings()
    seed_out_dir = Path(out_dir)
    seed_out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = seed_out_dir / seed_csv_name(seed)
    worker_output = io.StringIO()

    try:
        with contextlib.redirect_stdout(worker_output), contextlib.redirect_stderr(
            worker_output
        ):
            rows = run_experiment(
                seed=int(seed),
                n_samples=int(n_samples),
                model_type=str(model_type),
            )
            save_results_csv(rows, str(csv_path))
    except Exception as exc:
        captured_output = worker_output.getvalue().strip()
        details = f"\nWorker output:\n{captured_output}" if captured_output else ""
        raise RuntimeError(f"Seed {int(seed)} failed: {exc}{details}") from exc
    return str(csv_path)


def _read_seed_rows(seed_paths: Mapping[int, Path]) -> List[Dict[str, float]]:
    all_rows: List[Dict[str, float]] = []
    reference_alphas: tuple[float, ...] | None = None

    for expected_seed, path in sorted(seed_paths.items()):
        with path.open("r", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            fields = set(reader.fieldnames or [])
            missing = set(PER_SEED_COLUMNS) - fields
            if missing:
                raise ValueError(f"{path} is missing columns: {sorted(missing)}")

            seed_rows: List[Dict[str, float]] = []
            for line_number, raw_row in enumerate(reader, start=2):
                try:
                    row = {name: float(raw_row[name]) for name in PER_SEED_COLUMNS}
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"{path}:{line_number} contains a non-numeric result."
                    ) from exc

                if not all(np.isfinite(value) for value in row.values()):
                    raise ValueError(f"{path}:{line_number} contains a non-finite value.")
                if not float(row["seed"]).is_integer() or int(row["seed"]) != expected_seed:
                    raise ValueError(
                        f"{path}:{line_number} has seed={row['seed']}, "
                        f"expected {expected_seed}."
                    )
                if not np.isclose(row["target_coverage"], 1.0 - row["alpha"]):
                    raise ValueError(
                        f"{path}:{line_number} has inconsistent target coverage."
                    )
                seed_rows.append(row)

        if not seed_rows:
            raise ValueError(f"{path} contains no result rows.")

        alphas = tuple(sorted(row["alpha"] for row in seed_rows))
        if len(set(alphas)) != len(alphas):
            raise ValueError(f"{path} contains duplicate alpha rows.")
        if reference_alphas is None:
            reference_alphas = alphas
        elif alphas != reference_alphas:
            raise ValueError(
                f"{path} uses alpha grid {alphas}, expected {reference_alphas}."
            )
        all_rows.extend(seed_rows)

    return sorted(all_rows, key=lambda row: (int(row["seed"]), row["alpha"]))


def aggregate_by_alpha(rows: Sequence[Mapping[str, float]]) -> List[Dict[str, float]]:
    if not rows:
        raise ValueError("No per-seed rows found for aggregation.")

    grouped: Dict[float, List[Mapping[str, float]]] = {}
    for row in rows:
        grouped.setdefault(float(row["alpha"]), []).append(row)

    aggregated: List[Dict[str, float]] = []
    for alpha in sorted(grouped):
        group_rows = grouped[alpha]
        seeds = {int(row["seed"]) for row in group_rows}
        if len(seeds) != len(group_rows):
            raise ValueError(f"alpha={alpha} contains duplicate seed rows.")

        out_row: Dict[str, float] = {
            "alpha": float(alpha),
            "n_runs": float(len(group_rows)),
        }
        for metric in SUMMARY_METRICS:
            values = np.asarray([float(row[metric]) for row in group_rows], dtype=float)
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
            out_row[metric] = mean
            out_row[f"{metric}_std"] = std
            out_row[f"{metric}_se"] = float(std / np.sqrt(values.size))
        aggregated.append(out_row)
    return aggregated


def _atomic_write_csv(
    rows: Sequence[Mapping[str, Any]],
    destination: Path,
    fieldnames: Sequence[str],
) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {destination}.")
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(destination)


def save_summary_csv(rows: Sequence[Mapping[str, float]], destination: Path) -> None:
    mean_rows = [
        {column: float(row[column]) for column in SUMMARY_CSV_COLUMNS}
        for row in rows
    ]
    _atomic_write_csv(mean_rows, destination, SUMMARY_CSV_COLUMNS)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    seeds = validate_args(args)
    max_workers = min(int(args.max_workers), len(seeds))

    base_out_dir = Path(args.out_dir).resolve()
    out_dir = Path(build_model_out_dir(str(base_out_dir), str(args.model_type))).resolve()
    validate_output_directory(out_dir, seeds)
    out_dir.mkdir(parents=True, exist_ok=True)
    import tempfile
    matplotlib_config_dir = tempfile.TemporaryDirectory(prefix="simulation_final_matplotlib_")
    os.environ["MPLCONFIGDIR"] = matplotlib_config_dir.name

    print("Multi-seed simulation")
    print(f"seeds={seeds}")
    print(f"n_samples_per_seed={int(args.n_samples)}")
    print(f"model_type={args.model_type}")
    print(f"max_workers={max_workers}")
    print(f"output_dir={out_dir}")

    completed: Dict[int, Path] = {}
    failures: Dict[int, str] = {}
    spawn_context = get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=spawn_context,
    ) as executor:
        future_to_seed = {
            executor.submit(
                _run_seed,
                seed,
                str(out_dir),
                int(args.n_samples),
                str(args.model_type),
            ): seed
            for seed in seeds
        }
        for future in as_completed(future_to_seed):
            seed = future_to_seed[future]
            try:
                csv_path = Path(future.result())
                if not csv_path.exists():
                    raise RuntimeError(f"Worker did not create {csv_path}.")
                completed[seed] = csv_path
                print(f"[ok] seed={seed} -> {csv_path.name}", flush=True)
            except Exception as exc:
                failures[seed] = str(exc)
                print(
                    f"[failed] seed={seed}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )

    if failures:
        failure_summary = "; ".join(
            f"seed {seed}: {message}" for seed, message in sorted(failures.items())
        )
        raise RuntimeError(f"One or more seed runs failed: {failure_summary}")
    if set(completed) != set(seeds):
        raise RuntimeError("Completed seed set does not match the requested seed set.")

    all_rows = _read_seed_rows(completed)
    summary_rows = aggregate_by_alpha(all_rows)

    summary_csv = out_dir / "simulation_summary.csv"
    save_summary_csv(summary_rows, summary_csv)
    print_two_tables(summary_rows)
    save_plots(summary_rows, str(out_dir), str(args.model_type))

    print()
    print(f"Saved summary statistics: {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


