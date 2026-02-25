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


def expected_horizon_cost(net, r: float, start_step: int, horizon: int, curing_strategy="gradient"):
    """
    Run Simulation Until the End of the Horizon with Uniform Budget Spend
    """
    cost = 0.0
    for t in range(start_step, horizon + 1):
        cost += net.get_network_metrics()[1]  # S_bar
        b = min(even_spend(r, horizon - t + 1), r)
        simulate_with_budget(net, b, curing_strategy=curing_strategy)
        r -= b
        if r < 1e-12:
            r = 0.0
    return cost

def current_step_b(net_root,  r: float, step_idx: int, horizon: int, delta: float, n_rollouts: int, curing_strategy="gradient", base_seed: int = 123) -> float:
    """
    Simulates the Optimal Choice of Budget at Current Step
    """
    # Discrete Candidates up to the Remaining Budget
    max_k = int(r // delta)
    candidates = [k * delta for k in range(max_k + 1)]

    # Determine the Optimal Budget Allocation at this step
    best_b, best_val = 0.0, float("inf")
    for b in candidates:
        total = 0.0
        for j in range(n_rollouts):
            # Change Seed for Randomness
            seed = base_seed + 10_000 * step_idx + j
            np.random.seed(seed)
            random.seed(seed)

            # Copy initial network
            net = copy.deepcopy(net_root)
            cost = net.get_network_metrics()[1] # S_n

            # Apply Gradient Descent for Candidate and Simulate Expected Cost Until Horizon
            simulate_with_budget(net, b, curing_strategy=curing_strategy)
            r1 = r - b
            if step_idx < horizon:
                cost += expected_horizon_cost(net, r1, step_idx + 1, horizon, curing_strategy=curing_strategy)
            total += cost
        avg = total / n_rollouts
        if avg < best_val:
            best_val = avg
            best_b = b
    return best_b

def budget_plan(network, budget: float, delta: float, horizon: int, curing_strategy: str, n_rollouts: int, base_seed: int = 123) -> list[float]:
    """
    Returns Allocation Plan for the Time Horizon
    """
    r = float(budget)
    plan = []
    net = copy.deepcopy(network)

    for t in range(1, horizon + 1):
        # Choose b at this step
        b = current_step_b(net, r, step_idx=t, horizon=horizon, delta=delta, n_rollouts=n_rollouts, curing_strategy=curing_strategy, base_seed=base_seed)
        b = min(b, r)
        plan.append(b)
        # Advance to Next Step
        simulate_with_budget(net, b, curing_strategy=curing_strategy)
        r -= b
        if r < 1e-12:
            r = 0.0

    return plan