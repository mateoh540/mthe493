import numpy as np
import pandas as pd
from typing import Dict, Optional


def evaluate_metrics(
    S_bar: np.ndarray,
    extra_curing_used: np.ndarray,
    alpha: float = 0.10,
) -> Dict[str, float]:
    """
    Compute evaluation metrics for a single simulation run.

    Parameters
    ----------
    S_bar : np.ndarray
        Network exposure over time for one run.
    extra_curing_used : np.ndarray
        Curing cost used at each time step for one run.
    alpha : float
        Safe threshold for tau_alpha.

    Returns
    -------
    Dict[str, float]
        Dictionary containing I_peak, tau_alpha, B_burden, and C_cost.
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
    """
    Compute efficiency eta = (baseline_burden - B_burden) / C_cost.
    """
    if metrics["C_cost"] == 0.0:
        return 0.0
    return (baseline_burden - metrics["B_burden"]) / metrics["C_cost"]


def evaluate_all_strategies(
    all_simulation_data: Dict[str, np.ndarray],
    all_curing_costs: Optional[Dict[str, np.ndarray]] = None,
    alpha: float = 0.10,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate all curing strategies by averaging metrics across iterations.

    Parameters
    ----------
    all_simulation_data : dict
        Dictionary where all_simulation_data[strategy] has shape
        (iterations, num_steps, 2), with [:, :, 1] = S_bar.
    all_curing_costs : dict, optional
        Dictionary where all_curing_costs[strategy] has shape
        (iterations, num_steps). If omitted, zero costs are assumed.
    alpha : float
        Safe threshold for tau_alpha.

    Returns
    -------
    Dict[str, Dict[str, float]]
        Nested dictionary: results[strategy][metric] = averaged value.
    """
    results = {}
    baseline_burden = None

    for strategy, simulation_data in all_simulation_data.items():
        iterations, num_steps, _ = simulation_data.shape
        S_bar_all = simulation_data[:, :, 1]

        if all_curing_costs is not None and strategy in all_curing_costs:
            extra_curing_all = all_curing_costs[strategy]
        else:
            extra_curing_all = np.zeros((iterations, num_steps))

        all_metrics = {
            "I_peak": [],
            "tau_alpha": [],
            "B_burden": [],
            "C_cost": [],
        }

        for i in range(iterations):
            metrics_i = evaluate_metrics(
                S_bar=S_bar_all[i],
                extra_curing_used=extra_curing_all[i],
                alpha=alpha,
            )
            for key, value in metrics_i.items():
                all_metrics[key].append(value)

        avg_metrics = {}
        for key, values in all_metrics.items():
            if key == "tau_alpha":
                finite_values = [v for v in values if np.isfinite(v)]
                avg_metrics[key] = float(np.mean(finite_values)) if finite_values else np.inf
            else:
                avg_metrics[key] = float(np.mean(values))

        results[strategy] = avg_metrics

        if strategy == "none":
            baseline_burden = avg_metrics["B_burden"]

    if baseline_burden is not None:
        for strategy in results:
            results[strategy]["eta"] = compute_efficiency(
                results[strategy], baseline_burden
            )
    else:
        for strategy in results:
            results[strategy]["eta"] = np.nan

    return results


def summarize_results(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Convert nested results dictionary into a tidy DataFrame.
    """
    rows = []
    for strategy, metrics in results.items():
        rows.append({
            "strategy": strategy,
            **metrics,
        })
    return pd.DataFrame(rows)
    