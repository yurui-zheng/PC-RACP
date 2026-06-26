from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np

UTILITY_TABLE = np.array([
    [0.70, 0.60, 0.35, 0.15],
    [0.95, 0.55, 0.45, 0.15],
    [0.80, 0.50, 0.20, 0.20],
], dtype=float)


U_MAX: float = 1.0


def utility_func(a, y) -> float:
    a = int(a)
    y = int(y)
    return float(UTILITY_TABLE[a, y])


@dataclass
class LinearSoftmaxEnvironment:
    """
    A simple synthetic environment:

      X ~ N(0, I_d)
      A ~ pi_b(A|X)  (behavior policy; softmax over linear logits)
      Y ~ P(Y|X,A)   (outcome mechanism; softmax over linear logits)

    Stores the true parameters for counterfactual policy evaluation,
    including P(Y|X, a) for any action a.
    """
    d: int
    num_actions: int
    num_labels: int
    V_pref: np.ndarray   # (num_actions, d)
    c_pref: np.ndarray   # (num_actions,)
    W_out: np.ndarray    # (num_actions, num_labels, d)
    b_out: np.ndarray    # (num_actions, num_labels)
    rng: np.random.Generator

    def sample_X(self, n: int) -> np.ndarray:
        return self.rng.normal(size=(n, self.d))

    def behavior_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        logits = X @ self.V_pref.T + self.c_pref
        logits = logits / np.sqrt(self.d)
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        return exp_logits / exp_logits.sum(axis=1, keepdims=True)

    def sample_A(self, X: np.ndarray) -> np.ndarray:
        probs = self.behavior_proba(X)
        A = np.empty(X.shape[0], dtype=int)
        for i in range(X.shape[0]):
            A[i] = self.rng.choice(self.num_actions, p=probs[i])
        return A

    def outcome_proba(self, x: np.ndarray, a: int) -> np.ndarray:
        x = np.asarray(x).reshape(-1)
        a = int(a)
        logits_y = self.W_out[a] @ x + self.b_out[a]
        logits_y = logits_y * 1.2
        logits_y = logits_y - logits_y.max()
        exp_y = np.exp(logits_y)
        return exp_y / exp_y.sum()

    def sample_Y_given_A(self, X: np.ndarray, A: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        A = np.asarray(A).astype(int)
        Y = np.empty(X.shape[0], dtype=int)
        for i in range(X.shape[0]):
            p = self.outcome_proba(X[i], A[i])
            Y[i] = self.rng.choice(self.num_labels, p=p)
        return Y


def make_environment(
    random_state: int | None = 0,
    d: int = 10,
    num_actions: int = 3,
    num_labels: int = 4,
) -> LinearSoftmaxEnvironment:
    rng = np.random.default_rng(random_state)

    V_pref = rng.normal(loc=0.0, scale=1.0, size=(num_actions, d))
    c_pref = rng.normal(loc=0.0, scale=0.5, size=(num_actions,))

    W_out = rng.normal(loc=0.0, scale=1.0, size=(num_actions, num_labels, d))
    b_out = rng.normal(loc=0.0, scale=0.5, size=(num_actions, num_labels))
    W_out = W_out / np.sqrt(d)

    return LinearSoftmaxEnvironment(
        d=d,
        num_actions=num_actions,
        num_labels=num_labels,
        V_pref=V_pref,
        c_pref=c_pref,
        W_out=W_out,
        b_out=b_out,
        rng=rng,
    )


def generate_data(
    n_samples: int,
    random_state: int | None = 0,
    d: int = 10,
    num_actions: int = 3,
    num_labels: int = 4,
    return_env: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, np.ndarray, LinearSoftmaxEnvironment]:
    """
    Backward-compatible generator: returns (X, Y, A) by default.
    If return_env=True, returns (X, Y, A, env) where env stores the true parameters.
    """
    assert num_actions == 3, "This environment assumes A={0,1,2}"
    assert num_labels == 4, "This environment assumes Y={0,1,2,3}"

    env = make_environment(random_state=random_state, d=d, num_actions=num_actions, num_labels=num_labels)
    X = env.sample_X(n_samples)
    A = env.sample_A(X)
    Y = env.sample_Y_given_A(X, A)

    if return_env:
        return X, Y, A, env
    return X, Y, A
