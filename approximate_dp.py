# dp_planner.py
import copy
import random
import numpy as np


def simulate_with_budget(net, b: float, curing_strategy: str = "gradient"):
    """
    Simulates One Step of the Network with Budget b and Curing Strategy
    """
    net.set_budget(float(b))
    net.simulate_step(curing_strategy=curing_strategy)

def even_spend(r: float, steps_left: int) -> float:
    """
    Calculates the Unfirorm Allocation of Remaining Budget
    """
    return 0.0 if steps_left <= 0 else r / steps_left


def expected_horizon_cost(net, r: float, total_budget: float, start_step: int, horizon: int,
                          curing_strategy="gradient", alpha: float = 10):    
    cost = 0.0
    for t in range(start_step, horizon + 1):
        b = min(even_spend(r, horizon - t + 1), r)
        simulate_with_budget(net, b, curing_strategy=curing_strategy)

        cost += stage_cost(net, total_budget, alpha=alpha)

        r -= b
        if r < 1e-12:
            r = 0.0

    return cost


def stage_cost(net, total_budget: float, alpha: float) -> float:
    _, S_bar, wasted_budget = net.get_network_metrics()
    return S_bar + alpha * (wasted_budget / total_budget) if total_budget > 1e-12 else S_bar

def current_step_b(net_root, r: float, total_budget: float, step_idx: int, horizon: int, delta: float,
                   n_rollouts: int, curing_strategy="gradient",
                   base_seed: int = 123, alpha: float = 2) -> float:
    max_k = int(r // delta)
    candidates = [k * delta for k in range(max_k + 1)]

    best_b, best_val = 0.0, float("inf")
    for b in candidates:
        total = 0.0
        for j in range(n_rollouts):
            seed = base_seed + 10_000 * step_idx + j
            np.random.seed(seed)
            random.seed(seed)

            net = copy.deepcopy(net_root)

            simulate_with_budget(net, b, curing_strategy=curing_strategy)
            cost = stage_cost(net, total_budget, alpha=alpha)

            r1 = r - b
            if step_idx < horizon:
                cost += expected_horizon_cost(
                    net, r1, total_budget, step_idx + 1, horizon,
                    curing_strategy=curing_strategy,
                    alpha=alpha
                )

            total += cost

        avg = total / n_rollouts
        if avg < best_val:
            best_val = avg
            best_b = b

    return best_b

def budget_plan(network, budget: float, delta: float, horizon: int,
                curing_strategy: str, n_rollouts: int,
                base_seed: int = 123, alpha: float = 1.0) -> list[float]:
    """
    Returns allocation plan for the time horizon.
    """
    r = float(budget)
    plan = []
    net = copy.deepcopy(network)

    for t in range(1, horizon + 1):
        b = current_step_b(
            net, r,
            total_budget=budget,
            step_idx=t,
            horizon=horizon,
            delta=delta,
            n_rollouts=n_rollouts,
            curing_strategy=curing_strategy,
            base_seed=base_seed,
            alpha=alpha
        )
        b = min(b, r)
        plan.append(b)

        simulate_with_budget(net, b, curing_strategy=curing_strategy)
        r -= b
        if r < 1e-12:
            r = 0.0

    return plan