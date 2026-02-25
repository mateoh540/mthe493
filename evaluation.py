"""
Evaluation metrics for curing strategies on Polya networks.

Five metrics per strategy:
  1. I_peak = max(S_bar)               — peak misinformation prevalence
  2. tau_alpha = inf{n: S_bar(n)≤α}    — time to reach safe threshold α
  3. B_burden = sum(S_bar)             — cumulative exposure (area-under-curve)
  4. C_cost = sum(extra_curing_used)   — total intervention cost
  5. eta = (B_baseline - B_strategy)/C — efficiency (burden reduction per cost)

All computed on S̄(n) = network-wide average exposure S_bar[i,n] per iteration.
"""

import numpy as np
from typing import Dict, Optional
import pandas as pd


def evaluate_metrics(
    S_bar: np.ndarray,
    extra_curing_used: np.ndarray,
    alpha: float = 0.10
) -> Dict[str, float]:
    """
    Compute metrics from single trajectory.

    S_bar : network exposure over time
    extra_curing_used : cost of curing at each step
    alpha : safe threshold (default 0.10)
    """
    S_bar = np.asarray(S_bar, dtype=float)
    extra_curing_used = np.asarray(extra_curing_used, dtype=float)

    I_peak = float(np.max(S_bar))
    safe_idx = np.where(S_bar <= alpha)[0]
    tau_alpha = float(safe_idx[0]) if len(safe_idx) > 0 else np.inf
    B_burden = float(np.sum(S_bar))
    C_cost = float(np.sum(extra_curing_used))

    return {
        "I_peak": I_peak,
        "tau_alpha": tau_alpha,
        "B_burden": B_burden,
        "C_cost": C_cost,
    }


def compute_efficiency(metrics: Dict[str, float], baseline_burden: float) -> float:
    """Efficiency: (baseline_burden - B_burden) / C_cost."""
    if metrics["C_cost"] == 0.0:
        return 0.0
    return (baseline_burden - metrics["B_burden"]) / metrics["C_cost"]


def run_evaluation_pipeline(
    simulation_data: np.ndarray,
    curing_costs: Optional[Dict[str, np.ndarray]] = None,
    alpha: float = 0.10,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate all strategies; average metrics across iterations.

    simulation_data : shape (iterations, num_steps, 2); [:,:,1] = S̄(n)
    curing_costs : dict mapping strategy name → (iterations, num_steps) cost array
    alpha : safe threshold

    Returns
    -------
    dict of dicts: results[strategy][metric] = averaged_value
    Includes eta computed relative to 'none' strategy baseline.
    """
    iterations, num_steps, _ = simulation_data.shape
    S_bar_all = simulation_data[:, :, 1]

    default_zero = np.zeros((iterations, num_steps))
    strategies = list(curing_costs.keys()) if curing_costs else ["none"]

    results = {}
    baseline_burden = None

    for strategy in strategies:
        extra_curing = (
            curing_costs[strategy]
            if curing_costs and strategy in curing_costs
            else default_zero
        )

        all_metrics = {"I_peak": [], "tau_alpha": [], "B_burden": [], "C_cost": []}

        for i in range(iterations):
            metrics_i = evaluate_metrics(S_bar_all[i], extra_curing[i], alpha=alpha)
            for key, val in metrics_i.items():
                all_metrics[key].append(val)

        avg_metrics = {}
        for key, vals in all_metrics.items():
            if key == "tau_alpha":
                finite = [v for v in vals if np.isfinite(v)]
                avg_metrics[key] = float(np.mean(finite)) if finite else np.inf
            else:
                avg_metrics[key] = float(np.mean(vals))

        if strategy == "none":
            baseline_burden = avg_metrics["B_burden"]

        results[strategy] = avg_metrics

    if baseline_burden is not None:
        for strategy in strategies:
            results[strategy]["eta"] = compute_efficiency(
                results[strategy], baseline_burden
            )

    return results


def summarize_runs(results: Dict[str, Dict[str, Dict[str, float]]]) -> pd.DataFrame:
    """
    Flatten multi-condition results to DataFrame.

    results : dict where results[condition][strategy][metric] = value
    Returns : pd.DataFrame with columns [condition, strategy, I_peak, tau_alpha, B_burden, C_cost, eta]
    """
    rows = []
    for condition, strategies_dict in results.items():
        for strategy, metrics in strategies_dict.items():
            rows.append({
                "condition": condition,
                "strategy": strategy,
                **metrics,
            })
    return pd.DataFrame(rows)
