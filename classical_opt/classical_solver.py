"""
Classical Optimization Solver for Project ISOBAR
=================================================

This module provides the continuous baseline optimization solver for Project ISOBAR
(Inverse Risk-Weighted Asset Portfolio Shaping Engine).

Domain Rationale & Mathematical Specification
---------------------------------------------
1. Objective Function Definition:
   The classical optimization seeks an asset allocation vector X = [X_1, X_2, ..., X_N]^T
   that minimizes the total loss function f(X):

       f(X) = (sum_{i=1}^N X_i * RW_i - RWA_target)^2 + lambda * sum_{i=1}^N ((X_i - EAD_baseline_i) / EAD_baseline_i)^2

   - RWA Target Term: (sum(X_i * RW_i) - RWA_target)^2
     DESIGN DECISION: We explicitly use squared deviation z^2 where z = sum(X_i * RW_i) - RWA_target
     rather than absolute deviation |z|. Although |z| represents standard absolute L1 error,
     its non-differentiability at z = 0 creates a non-smooth objective surface with sharp gradient jumps,
     causing gradient-based numerical solvers (e.g., SLSQP) to chatter, stall, or fail to converge
     near the optimal solution. For a single-target formulation, z^2 and |z| share the exact same
     global minimizer X* when z = 0 is feasible, while z^2 provides a smooth C^2 objective function
     with well-defined gradients for robust convergence.

   - Disruption Penalty Term: lambda * sum(((X_i - EAD_baseline_i) / EAD_baseline_i)^2)
     Penalises relative percentage deviation of each asset's Exposure at Default (EAD)
     from its baseline origination value. The hyperparameter lambda (lambda_reg) controls
     the operational trade-off between achieving the exact regulatory RWA target and
     minimising portfolio re-balancing friction.

2. Decision Variables & Policy Bounds:
   For each asset bucket i:
       ead_min_limit_i <= X_i <= ead_max_limit_i
   These bounds reflect strict policy constraints (e.g., origination limits and credit policy floors).

3. Concentration Constraints (if enabled):
   For asset class C in portfolio:
       sum_{i in C} X_i / sum_{j=1}^N X_j <= max_share_C
   SciPy expects inequality constraints in standard form g(X) >= 0.
   Rearranging:
       max_share_C * sum_{j=1}^N X_j - sum_{i in C} X_i >= 0

4. Output Schema:
   The solver serializes its output to JSON conforming to `shared.schemas.SolverResult`.
"""

import sys
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Ensure /app is at the top of sys.path for Docker container mounting
sys.path.insert(0, '/app')

# Also include workspace root in sys.path for local non-container execution
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from shared.config_loader import load_config
from shared.schemas import SolverResult


def objective_function(
    X: np.ndarray,
    rw_vector: np.ndarray,
    ead_baseline: np.ndarray,
    target_rwa: float,
    lambda_reg: float
) -> float:
    """
    Compute scalar objective function value f(X).

    Parameters
    ----------
    X : np.ndarray
        Vector of proposed EAD values for each asset bucket.
    rw_vector : np.ndarray
        Vector of pre-calculated Risk Weights (RW_i) for each asset bucket.
    ead_baseline : np.ndarray
        Vector of baseline Exposure at Default values (EAD_baseline_i).
    target_rwa : float
        Target Risk-Weighted Assets amount (RWA_target).
    lambda_reg : float
        Regularisation weight balancing RWA targeting against balance sheet disruption.

    Returns
    -------
    float
        Calculated objective value f(X).
    """
    achieved_rwa = np.sum(X * rw_vector)
    rwa_term = (achieved_rwa - target_rwa) ** 2
    disruption_term = lambda_reg * np.sum(((X - ead_baseline) / ead_baseline) ** 2)
    return float(rwa_term + disruption_term) / 1e12


def jacobian_function(
    X: np.ndarray,
    rw_vector: np.ndarray,
    ead_baseline: np.ndarray,
    target_rwa: float,
    lambda_reg: float
) -> np.ndarray:
    """Analytical gradient of the scaled objective function."""
    achieved_rwa = np.sum(X * rw_vector)
    grad_rwa = 2.0 * (achieved_rwa - target_rwa) * rw_vector
    grad_disruption = 2.0 * lambda_reg * (X - ead_baseline) / (ead_baseline ** 2)
    return (grad_rwa + grad_disruption) / 1e12


def build_concentration_constraints(
    df: pd.DataFrame,
    concentration_config
) -> list[dict]:
    """
    Construct SciPy inequality constraints for asset class concentration limits.

    SciPy inequality constraints enforce ``fun(X) >= 0``.
    Given requirement: sum(X_i for i in asset_class) / sum(X) <= max_share
    Transformed into: max_share * sum(X) - sum(X_i for i in asset_class) >= 0

    Parameters
    ----------
    df : pd.DataFrame
        Asset register DataFrame containing 'asset_class' column.
    concentration_config : ConcentrationConfig
        Concentration constraint configuration from load_config().

    Returns
    -------
    list[dict]
        List of SciPy constraint dictionaries formatted for minimize(..., constraints=...).
    """
    constraints = []
    if not concentration_config.enabled:
        return constraints

    asset_classes = df["asset_class"].values

    for limit in concentration_config.limits:
        ac_name = limit.asset_class
        max_share = limit.max_share
        mask = (asset_classes == ac_name)

        # Bind mask and max_share as default parameters to avoid closure late-binding issues
        def con_fn(X, mask=mask, max_share=max_share):
            return max_share * np.sum(X) - np.sum(X[mask])

        constraints.append({
            "type": "ineq",
            "fun": con_fn
        })

    return constraints


def run_classical_solver() -> SolverResult:
    """
    Execute classical SLSQP optimization process and return SolverResult.

    Returns
    -------
    SolverResult
        Structured optimization result containing achieved RWA, disruption score,
        convergence status, wall-clock time, and final EAD vector.
    """
    config = load_config()

    # Determine input dataset path with fallback for local testing environments
    data_dir = Path(config.output.data_dir)
    csv_path = data_dir / "asset_register.csv"
    if not csv_path.exists():
        fallback_path = _PROJECT_ROOT / "data" / "assets" / "asset_register.csv"
        if fallback_path.exists():
            csv_path = fallback_path
        else:
            raise FileNotFoundError(
                f"Asset register CSV not found at '{csv_path}' or '{fallback_path}'."
            )

    df = pd.read_csv(csv_path)

    # Extract required numerical vectors
    rw_vector = df["precalc_rw"].to_numpy(dtype=float)
    ead_baseline = df["ead_baseline"].to_numpy(dtype=float)
    ead_min_limit = df["ead_min_limit"].to_numpy(dtype=float)
    ead_max_limit = df["ead_max_limit"].to_numpy(dtype=float)

    # Asset boundary limits (min, max)
    bounds = list(zip(ead_min_limit, ead_max_limit))

    # Initial guess X0 = ead_baseline vector
    X0 = ead_baseline.copy()

    # Optimization config parameters
    target_rwa = float(config.optimization.target_rwa)
    lambda_reg = float(config.optimization.lambda_reg)
    method = config.optimization.classical.method
    maxiter = config.optimization.classical.maxiter
    ftol = config.optimization.classical.ftol

    # Build concentration constraints if enabled
    constraints = build_concentration_constraints(
        df, config.optimization.concentration_constraints
    )

    # Solver options
    options = {
        "maxiter": maxiter,
        "ftol": ftol,
        "disp": False
    }

    start_time = time.perf_counter()
    res = minimize(
        fun=objective_function,
        x0=X0,
        args=(rw_vector, ead_baseline, target_rwa, lambda_reg),
        method=method,
        jac=jacobian_function,
        bounds=bounds,
        constraints=constraints if constraints else None,
        options=options
    )
    wall_clock_seconds = time.perf_counter() - start_time

    X_solved = res.x
    achieved_rwa = float(np.sum(X_solved * rw_vector))
    rwa_delta_pct = float(abs(achieved_rwa - target_rwa) / target_rwa * 100.0)
    disruption_score = float(np.sum(((X_solved - ead_baseline) / ead_baseline) ** 2))
    converged = bool(res.success)

    result = SolverResult(
        solver_type="classical",
        target_rwa=target_rwa,
        achieved_rwa=achieved_rwa,
        rwa_delta_pct=rwa_delta_pct,
        lambda_reg=lambda_reg,
        disruption_score=disruption_score,
        wall_clock_seconds=wall_clock_seconds,
        converged=converged,
        ead_vector=X_solved.tolist()
    )

    # Determine results output directory and create if needed
    results_dir = Path(config.output.results_dir)
    if not results_dir.exists() and not results_dir.parent.exists():
        results_dir = _PROJECT_ROOT / "data" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    output_file = results_dir / "classical_result.json"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(result.model_dump_json(indent=2))

    # Print summary of results
    print("=" * 60)
    print("      PROJECT ISOBAR — CLASSICAL SOLVER RESULTS      ")
    print("=" * 60)
    print(f"Solver Method       : {method}")
    print(f"Convergence Status  : {'CONVERGED' if converged else 'FAILED'}")
    print(f"Message             : {res.message}")
    print(f"Wall-Clock Time     : {wall_clock_seconds:.4f} seconds")
    print(f"Target RWA          : £{target_rwa:,.2f}")
    print(f"Achieved RWA        : £{achieved_rwa:,.2f}")
    print(f"RWA Absolute Delta  : £{abs(achieved_rwa - target_rwa):,.2f}")
    print(f"RWA Delta %         : {rwa_delta_pct:.4f}%")
    print(f"Disruption Score    : {disruption_score:.6f}")
    print(f"Results Saved To    : {output_file}")
    print("=" * 60)

    return result


if __name__ == "__main__":
    run_classical_solver()
