import copy
import random
import numpy as np

from simulation import Network  # imports your simulator classes


def Sbar(network) -> float:
    """Scalar state S_n = \tilde S_n (network exposure)."""
    return network.get_network_metrics()[1]  # (U_bar, S_bar)


def step_with_budget(network, b: float, curing_strategy: str = "gradient") -> float:
    """Advance one sim step with budget b; returns next S_bar."""
    network.set_budget(float(b))
    network.simulate_step(curing_strategy=curing_strategy)
    return Sbar(network)


def even_spend(r: float, steps_left: int) -> float:
    return 0.0 if steps_left <= 0 else r / steps_left


def rollout_tail_cost(network, r: float, start_step: int, horizon: int,
                      curing_strategy: str = "gradient") -> float:
    """Roll out with even spending; cost = sum of S_bar."""
    cost = 0.0
    for n in range(start_step, horizon + 1):
        cost += Sbar(network)
        b = min(even_spend(r, horizon - n + 1), r)
        step_with_budget(network, b, curing_strategy=curing_strategy)
        r -= b
        if r <= 1e-12:
            r = 0.0
    return cost


def choose_b_one_step_rollout(network, r: float, step_idx: int, horizon: int,
                             delta: float, n_rollouts: int,
                             curing_strategy: str = "gradient",
                             base_seed: int = 123) -> float:
    """One-step rollout DP: choose spend b now by simulating."""
    if r <= 0:
        return 0.0

    max_k = int(r // delta)
    candidates = [k * delta for k in range(max_k + 1)]

    best_b = 0.0
    best_val = float("inf")

    root = copy.deepcopy(network)

    for b in candidates:
        total = 0.0
        for j in range(n_rollouts):
            # Common random numbers across candidates
            seed = base_seed + 10_000 * step_idx + j
            np.random.seed(seed)
            random.seed(seed)

            net = copy.deepcopy(root)

            cost = Sbar(net)  # cost at time n
            step_with_budget(net, b, curing_strategy=curing_strategy)
            r1 = r - b

            if step_idx < horizon:
                cost += rollout_tail_cost(
                    net, r1, start_step=step_idx + 1, horizon=horizon,
                    curing_strategy=curing_strategy
                )
            total += cost

        avg = total / n_rollouts
        if avg < best_val:
            best_val = avg
            best_b = b

    return best_b


def run_episode_rollout_dp(network, total_budget: float, horizon: int = 5,
                           delta: float | None = None, n_rollouts: int = 10,
                           curing_strategy: str = "gradient",
                           base_seed: int = 123):
    """Run controlled episode; returns spends and S_bar trajectory."""
    B = float(total_budget)
    if delta is None:
        delta = B / 20  # ~21 actions

    r = B
    spends = []
    S_path = [Sbar(network)]

    for n in range(1, horizon + 1):
        b = choose_b_one_step_rollout(
            network, r, step_idx=n, horizon=horizon,
            delta=delta, n_rollouts=n_rollouts,
            curing_strategy=curing_strategy, base_seed=base_seed
        )
        b = min(b, r)
        step_with_budget(network, b, curing_strategy=curing_strategy)
        r -= b
        spends.append(b)
        S_path.append(Sbar(network))

    return spends, S_path


if __name__ == "__main__":
    # Example run
    net = Network()
    net.initialize_barabasi_albert(n=100, m=1, initial_conditions=[5, 5])

    B_total = len(net.nodes) * 5
    spends, S_path = run_episode_rollout_dp(
        net, total_budget=B_total, horizon=5, delta=B_total/20, n_rollouts=10
    )

    print("Chosen spends:", spends)
    print("Total spend:", sum(spends))
    print("S path:", S_path)