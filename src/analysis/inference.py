"""Dependency-light, deterministic econometric inference primitives.

All routines are pure and use NumPy only.  Callers are responsible for fitting
the point estimator and passing its design matrix and residuals in matching row
order.  The standard-error routines return the square root of the diagonal of
the corresponding sandwich covariance matrix.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeVar

import numpy as np


ArrayLike = Sequence[float] | np.ndarray
T = TypeVar("T")


class SmallTError(ValueError):
    """Raised when Driscoll--Kraay asymptotics are not credible for the panel."""


def _reject_missing_ids(values: list, kind: str) -> None:
    """Raise if any identifier is null/NaN/NaT/non-finite — such an id defines no
    valid group (NaN != NaN) and would silently drop its observation."""
    for value in values:
        invalid = value is None
        if isinstance(value, (float, complex, np.floating, np.complexfloating)):
            invalid = invalid or not bool(np.isfinite(value))
        elif isinstance(value, (np.datetime64, np.timedelta64)):
            invalid = invalid or bool(np.isnat(value))
        elif callable(getattr(value, "is_finite", None)):
            invalid = invalid or not bool(value.is_finite())
        else:
            try:
                invalid = invalid or not bool(value == value)
            except (TypeError, ValueError):
                invalid = True
        if invalid:
            raise ValueError(f"{kind} must be non-null and finite")


def _regression_arrays(residuals: ArrayLike, X: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    u = np.asarray(residuals, dtype=float).reshape(-1)
    design = np.asarray(X, dtype=float)
    if design.ndim != 2:
        raise ValueError("X must be a two-dimensional design matrix")
    if len(u) != len(design):
        raise ValueError("residuals and X must have the same number of rows")
    if len(u) == 0 or design.shape[1] == 0:
        raise ValueError("residuals and X must be non-empty")
    if not np.isfinite(u).all() or not np.isfinite(design).all():
        raise ValueError("residuals and X must contain only finite values")
    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("X must have full column rank")
    return u, design


def _sandwich_se(design: np.ndarray, meat: np.ndarray) -> np.ndarray:
    # Sandwich covariance: (X'X)^-1 B (X'X)^-1.
    bread = np.linalg.inv(design.T @ design)
    covariance = bread @ meat @ bread
    diagonal = np.diag((covariance + covariance.T) / 2.0)
    # Floating point cancellation can create harmless values just below zero.
    if np.any(diagonal < -1e-12):
        raise ValueError("estimated covariance has a negative diagonal")
    return np.sqrt(np.maximum(diagonal, 0.0))


def newey_west_se(residuals: ArrayLike, X: ArrayLike, lags: int) -> np.ndarray:
    """Return Bartlett-kernel HAC/Newey--West standard errors.

    With score ``z_t = x_t u_t`` and Bartlett weight
    ``w_l = 1 - l/(L+1)``, the meat is
    ``B = sum_t z_t z_t' + sum_l w_l sum_t(z_t z_{t-l}' + z_{t-l} z_t')``.
    No finite-sample multiplier is applied.
    """

    u, design = _regression_arrays(residuals, X)
    if not isinstance(lags, int) or isinstance(lags, bool) or lags < 0 or lags >= len(u):
        raise ValueError("lags must be an integer in [0, n - 1]")
    scores = design * u[:, None]
    meat = scores.T @ scores
    for lag in range(1, lags + 1):
        gamma = scores[lag:].T @ scores[:-lag]
        weight = 1.0 - lag / (lags + 1.0)
        meat += weight * (gamma + gamma.T)
    return _sandwich_se(design, meat)


def driscoll_kraay_se(
    residuals: ArrayLike,
    X: ArrayLike,
    time_ids: Sequence[object] | np.ndarray,
    lags: int,
    *,
    min_time_periods: int = 20,
) -> np.ndarray:
    """Return Driscoll--Kraay standard errors, refusing panels with small ``T``.

    The cross-sectional score is ``S_t = sum_i x_it u_it``.  The covariance
    meat is Newey--West HAC applied to ``S_t`` with Bartlett weights, while the
    bread remains ``(X'X)^-1``.  DK relies on large-T asymptotics, so fewer than
    ``min_time_periods`` distinct periods raises :class:`SmallTError`.
    """

    u, design = _regression_arrays(residuals, X)
    times = np.asarray(time_ids, dtype=object).reshape(-1)
    if len(times) != len(u):
        raise ValueError("time_ids and X must have the same number of rows")
    if not isinstance(min_time_periods, int) or min_time_periods < 2:
        raise ValueError("min_time_periods must be an integer of at least 2")
    # Reject null/NaN/NaT time IDs BEFORE dedup — a missing period would satisfy
    # the large-T guard yet match no observation (NaN != NaN), silently dropping
    # its score and corrupting the DK meat and the guard (Codex R2-1).
    _reject_missing_ids(times.tolist(), "time identifiers")
    unique_times = list(dict.fromkeys(times.tolist()))
    try:
        unique_times.sort()
    except TypeError as exc:
        raise ValueError("time_ids must be chronologically comparable") from exc
    periods = len(unique_times)
    if periods < min_time_periods:
        raise SmallTError(
            f"Driscoll-Kraay requires at least {min_time_periods} time periods; got {periods}"
        )
    if not isinstance(lags, int) or isinstance(lags, bool) or lags < 0 or lags >= periods:
        raise ValueError("lags must be an integer in [0, T - 1]")

    scores = design * u[:, None]
    by_time = np.vstack([scores[times == period].sum(axis=0) for period in unique_times])
    meat = by_time.T @ by_time
    for lag in range(1, lags + 1):
        gamma = by_time[lag:].T @ by_time[:-lag]
        weight = 1.0 - lag / (lags + 1.0)
        meat += weight * (gamma + gamma.T)
    return _sandwich_se(design, meat)


def _cluster_covariance(
    design: np.ndarray,
    residuals: np.ndarray,
    cluster_ids: Sequence[object] | np.ndarray,
) -> np.ndarray:
    clusters = np.asarray(cluster_ids, dtype=object).reshape(-1)
    if len(clusters) != len(residuals):
        raise ValueError("cluster_ids and X must have the same number of rows")
    cluster_values = clusters.tolist()
    _reject_missing_ids(cluster_values, "cluster identifiers")
    try:
        unique_clusters = list(dict.fromkeys(cluster_values))
    except TypeError as exc:
        raise ValueError("cluster identifiers must be hashable") from exc
    if len(unique_clusters) < 2:
        raise ValueError("clustered inference requires at least two clusters")
    scores = design * residuals[:, None]
    meat = np.zeros((design.shape[1], design.shape[1]), dtype=float)
    memberships = np.zeros(len(cluster_values), dtype=int)
    for cluster in unique_clusters:
        try:
            mask = np.fromiter(
                (bool(value == cluster) for value in cluster_values),
                dtype=bool,
                count=len(cluster_values),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("cluster identifiers must support scalar equality") from exc
        memberships += mask.astype(int)
        cluster_score = scores[mask].sum(axis=0)
        meat += np.outer(cluster_score, cluster_score)
    if not np.all(memberships == 1):
        raise ValueError("every observation must belong to exactly one cluster")
    bread = np.linalg.inv(design.T @ design)
    return bread @ meat @ bread


def clustered_se(
    X: ArrayLike,
    residuals: ArrayLike,
    cluster_ids: Sequence[object] | np.ndarray,
) -> np.ndarray:
    """Return one-way Liang--Zeger cluster-robust standard errors.

    For cluster score ``S_g = sum_{i in g} x_i u_i``, the covariance is
    ``(X'X)^-1 [sum_g S_g S_g'] (X'X)^-1``.  No CR1 correction is applied.
    """

    u, design = _regression_arrays(residuals, X)
    covariance = _cluster_covariance(design, u, cluster_ids)
    diagonal = np.diag((covariance + covariance.T) / 2.0)
    return np.sqrt(np.maximum(diagonal, 0.0))


def two_way_clustered_se(
    X: ArrayLike,
    residuals: ArrayLike,
    cluster_ids_a: Sequence[object] | np.ndarray,
    cluster_ids_b: Sequence[object] | np.ndarray,
) -> np.ndarray:
    """Return Cameron--Gelbach--Miller two-way clustered standard errors.

    Inclusion--exclusion gives ``V_AB = V_A + V_B - V_{A intersection B}``,
    where every component is the uncorrected Liang--Zeger covariance.
    """

    u, design = _regression_arrays(residuals, X)
    a = np.asarray(cluster_ids_a, dtype=object).reshape(-1)
    b = np.asarray(cluster_ids_b, dtype=object).reshape(-1)
    if len(a) != len(u) or len(b) != len(u):
        raise ValueError("both cluster id arrays must match X")
    intersections = np.empty(len(u), dtype=object)
    intersections[:] = list(zip(a.tolist(), b.tolist(), strict=True))
    covariance = (
        _cluster_covariance(design, u, a)
        + _cluster_covariance(design, u, b)
        - _cluster_covariance(design, u, intersections)
    )
    symmetric = (covariance + covariance.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    projected = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
    return np.sqrt(np.maximum(np.diag(projected), 0.0))


def _take_rows(data: T, indices: np.ndarray) -> T:
    if hasattr(data, "iloc"):
        return data.iloc[indices].reset_index(drop=True)  # type: ignore[no-any-return,union-attr]
    return np.asarray(data)[indices]  # type: ignore[return-value]


def block_bootstrap_ci(
    estimator_fn: Callable[[T], float],
    data: T,
    n_boot: int,
    block_len: int,
    seed: int,
    alpha: float,
) -> tuple[float, float, float]:
    """Return a deterministic moving-block bootstrap percentile interval.

    Blocks of length ``block_len`` are sampled uniformly from the ``n-L+1``
    overlapping blocks and concatenated, truncating the last block to ``n``.
    The interval is the empirical ``alpha/2`` and ``1-alpha/2`` quantiles.
    """

    n = len(data)  # type: ignore[arg-type]
    if n < 1:
        raise ValueError("data must be non-empty")
    if not isinstance(n_boot, int) or n_boot < 1:
        raise ValueError("n_boot must be a positive integer")
    if not isinstance(block_len, int) or block_len < 1 or block_len > n:
        raise ValueError("block_len must be an integer in [1, n]")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between zero and one")
    point = float(estimator_fn(data))
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    blocks_needed = (n + block_len - 1) // block_len
    for draw in range(n_boot):
        starts = rng.integers(0, n - block_len + 1, size=blocks_needed)
        indices = np.concatenate(
            [np.arange(start, start + block_len, dtype=int) for start in starts]
        )[:n]
        draws[draw] = float(estimator_fn(_take_rows(data, indices)))
    if not np.isfinite(draws).all() or not np.isfinite(point):
        raise ValueError("estimator_fn must return finite scalar values")
    lo, hi = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
    return point, float(lo), float(hi)


def bai_perron_breaks(
    series: ArrayLike,
    max_breaks: int,
    min_segment: int,
) -> tuple[list[int], float]:
    """Estimate piecewise-constant Bai--Perron-class breaks by dynamic programming.

    Segment cost is ``SSR(i,j) = sum_{t=i}^{j-1}(y_t - mean(y_i:j))^2``.
    Dynamic programming computes the minimum SSR for every feasible break count
    up to ``max_breaks``, subject to ``min_segment`` observations per segment.
    It then minimizes ``n*log(SSR/n) + (m+1)*log(n)`` (Gaussian BIC for the
    ``m+1`` segment means), preferring fewer breaks on ties.  Break indices are
    the first observation of each new segment.
    """

    values = np.asarray(series, dtype=float).reshape(-1)
    n = len(values)
    if n == 0 or not np.isfinite(values).all():
        raise ValueError("series must be non-empty and finite")
    if not isinstance(max_breaks, int) or isinstance(max_breaks, bool) or max_breaks < 0:
        raise ValueError("max_breaks must be a non-negative integer")
    if not isinstance(min_segment, int) or min_segment < 1:
        raise ValueError("min_segment must be a positive integer")
    if n < min_segment:
        raise ValueError("series is too short for min_segment")
    max_segments = min(max_breaks + 1, n // min_segment)

    prefix = np.concatenate(([0.0], np.cumsum(values)))
    prefix_sq = np.concatenate(([0.0], np.cumsum(values * values)))

    def cost(start: int, end: int) -> float:
        count = end - start
        total = prefix[end] - prefix[start]
        return float(max(prefix_sq[end] - prefix_sq[start] - total * total / count, 0.0))

    # SCALE-AWARE tolerances (Codex R2-2): an ABSOLUTE zero-SSR / DP tolerance
    # makes break-count selection depend on measurement units — a tiny-valued
    # series' genuinely-positive no-break SSR falls under an absolute threshold
    # and is misread as a perfect fit. Tie the tolerance to the total sum of
    # squares (no >=1 floor) so scaling the series does not change the answer.
    scale = float(np.dot(values, values))
    zero_tolerance = np.finfo(float).eps * scale * n

    infinity = float("inf")
    dp = np.full((max_segments + 1, n + 1), infinity)
    previous = np.full((max_segments + 1, n + 1), -1, dtype=int)
    dp[0, 0] = 0.0
    for count in range(1, max_segments + 1):
        earliest_end = count * min_segment
        for end in range(earliest_end, n + 1):
            start_min = (count - 1) * min_segment
            start_max = end - min_segment
            for start in range(start_min, start_max + 1):
                candidate = dp[count - 1, start] + cost(start, end)
                if candidate < dp[count, end] - zero_tolerance:
                    dp[count, end] = candidate
                    previous[count, end] = start

    selected_segments = 1
    selected_criterion = float("inf")
    for segments in range(1, max_segments + 1):
        ssr = float(max(dp[segments, n], 0.0))
        criterion = (
            float("-inf")
            if ssr <= zero_tolerance
            else n * float(np.log(ssr / n)) + segments * float(np.log(n))
        )
        if criterion < selected_criterion - 1e-12:
            selected_criterion = criterion
            selected_segments = segments

    breaks: list[int] = []
    end = n
    for count in range(selected_segments, 1, -1):
        start = int(previous[count, end])
        if start < 0:
            raise ValueError("no feasible break partition")
        breaks.append(start)
        end = start
    breaks.reverse()
    return breaks, float(max(dp[selected_segments, n], 0.0))


def oster_delta(
    beta_controlled: float,
    beta_uncontrolled: float,
    r2_controlled: float,
    r2_uncontrolled: float,
    r_max: float,
    *,
    beta_target: float = 0.0,
) -> float:
    """Return Oster's proportional-selection delta needed to reach ``beta_target``.

    Under the standard approximation,
    ``delta = (beta_c-beta*) (R_c-R_u) / [(beta_u-beta_c)(R_max-R_c)]``.
    """

    values = np.asarray(
        [beta_controlled, beta_uncontrolled, r2_controlled, r2_uncontrolled, r_max, beta_target],
        dtype=float,
    )
    if not np.isfinite(values).all():
        raise ValueError("Oster inputs must be finite")
    if not 0.0 <= r2_uncontrolled <= r2_controlled < r_max <= 1.0:
        raise ValueError("require 0 <= R2_uncontrolled <= R2_controlled < R_max <= 1")
    denominator = (beta_uncontrolled - beta_controlled) * (r_max - r2_controlled)
    if denominator == 0.0:
        raise ValueError("Oster delta is undefined for the supplied coefficients")
    return float(
        (beta_controlled - beta_target)
        * (r2_controlled - r2_uncontrolled)
        / denominator
    )


def cinelli_hazlett_rv(estimate: float, se: float, dof: int) -> float:
    """Return the Cinelli--Hazlett robustness value for reducing an estimate to zero.

    For ``f = |t|/sqrt(dof)``, equal-strength confounding in treatment and
    outcome has ``RV = (sqrt(f^4 + 4 f^2) - f^2)/2``.
    """

    if not np.isfinite(estimate) or not np.isfinite(se) or se <= 0.0:
        raise ValueError("estimate must be finite and se must be positive")
    if not isinstance(dof, int) or isinstance(dof, bool) or dof < 1:
        raise ValueError("dof must be a positive integer")
    f_value = abs(float(estimate) / float(se)) / np.sqrt(float(dof))
    f_squared = f_value * f_value
    return float((np.sqrt(f_squared * f_squared + 4.0 * f_squared) - f_squared) / 2.0)


__all__ = [
    "SmallTError",
    "bai_perron_breaks",
    "block_bootstrap_ci",
    "cinelli_hazlett_rv",
    "clustered_se",
    "driscoll_kraay_se",
    "newey_west_se",
    "oster_delta",
    "two_way_clustered_se",
]
