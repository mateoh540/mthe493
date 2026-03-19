import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
from collections import deque
import argparse
import pandas as pd
import json


# ==========================================================
# Node class
# ==========================================================
class Node:
    def __init__(
        self,
        node_id,
        initial_red=5,
        initial_black=5,
        delta_red=5,
        delta_black=5,
        memory_size=10,
    ):
        self.node_id = node_id
        self.urn_red = float(initial_red)
        self.urn_black = float(initial_black)
        self.delta_red = float(delta_red)
        self.delta_black = float(delta_black)
        self.memory_size = memory_size

        # stores tuples: (draw_result, delta_red_used, delta_black_used)
        self.draw_queue = deque()

    def get_proportion(self):
        total = self.urn_red + self.urn_black
        return 0.0 if total <= 0 else self.urn_red / total

    def update(self, draw_result, delta_black_step=None):
        """
        draw_result = 1 for red, 0 for black
        """
        if delta_black_step is None:
            delta_black_step = self.delta_black

        delta_black_step = float(max(0.0, delta_black_step))

        # store actual reinforcement used this step
        self.draw_queue.append((draw_result, self.delta_red, delta_black_step))

        # remove oldest if finite memory exceeded
        old = self.draw_queue.popleft() if len(self.draw_queue) > self.memory_size else None

        # apply new reinforcement
        if draw_result == 1:
            self.urn_red += self.delta_red
        else:
            self.urn_black += delta_black_step

        # undo oldest reinforcement if needed
        if old is not None:
            old_draw, old_dr, old_db = old
            if old_draw == 1:
                self.urn_red -= old_dr
            else:
                self.urn_black -= old_db

        self.urn_red = max(0.0, self.urn_red)
        self.urn_black = max(0.0, self.urn_black)


# ==========================================================
# BA Network class: undirected and unweighted only
# ==========================================================
class BANetwork:
    def __init__(self):
        self.nodes = {}
        self.graph = nx.Graph()
        self.budget_B = None
        self.cached_super_urn = {}

    def set_budget(self, B):
        self.budget_B = float(B)

    def initialize_barabasi_albert(self, n, m, initial_conditions, memory_size=10):
        """
        Create a plain undirected, unweighted BA graph.
        """
        ba_graph = nx.barabasi_albert_graph(n, m)

        delta_red, delta_black = initial_conditions
        self.graph = ba_graph

        for nid in self.graph.nodes():
            self.nodes[nid] = Node(
                node_id=nid,
                initial_red=5,
                initial_black=5,
                delta_red=delta_red,
                delta_black=delta_black,
                memory_size=memory_size,
            )

    # ------------------------------------------------------
    # Super urn
    # ------------------------------------------------------
    def _neighbor_closed(self, i):
        """
        Closed neighborhood N_i' = {i} U neighbors(i)
        """
        return [i] + list(self.graph.neighbors(i))

    def get_super_urn_proportion(self, node_id):
        """
        Since graph is unweighted and undirected, super urn is just
        the sum over node i and all its neighbors.
        """
        neigh = self._neighbor_closed(node_id)

        total_red = sum(self.nodes[j].urn_red for j in neigh)
        total_balls = sum(self.nodes[j].urn_red + self.nodes[j].urn_black for j in neigh)

        return 0.0 if total_balls <= 0 else total_red / total_balls

    def _compute_p(self):
        """
        Current p_i = S_{i,n-1}
        """
        return {i: self.get_super_urn_proportion(i) for i in self.nodes.keys()}

    # ------------------------------------------------------
    # Expected next-step approximation for GD
    # ------------------------------------------------------
    def _expected_red_black_after_step(self, p, x):
        red_exp = {}
        black_exp = {}

        for i, node in self.nodes.items():
            will_pop = len(node.draw_queue) >= node.memory_size
            popped = node.draw_queue[0] if will_pop else None

            add_red = node.delta_red * p[i]
            add_black = x[i] * (1.0 - p[i])

            rem_red = 0.0
            rem_black = 0.0

            if popped is not None:
                old_draw, old_dr, old_db = popped
                if old_draw == 1:
                    rem_red = float(old_dr)
                else:
                    rem_black = float(old_db)

            red_exp[i] = max(0.0, node.urn_red + add_red - rem_red)
            black_exp[i] = max(0.0, node.urn_black + add_black - rem_black)

        return red_exp, black_exp

    def _objective_expected_Sbar(self, p, x):
        """
        Approximate expected average network exposure after one step.
        """
        red_exp, black_exp = self._expected_red_black_after_step(p, x)

        total = 0.0
        for i in self.nodes.keys():
            neigh = self._neighbor_closed(i)
            num = sum(red_exp[j] for j in neigh)
            den = sum(red_exp[j] + black_exp[j] for j in neigh)
            total += (num / den) if den > 0 else 0.0

        return total / len(self.nodes)

    def _gradient_expected_Sbar(self, p, x):
        """
        Gradient of the approximate objective.
        """
        red_exp, black_exp = self._expected_red_black_after_step(p, x)

        num = {}
        den = {}
        closed_neigh = {}

        for i in self.nodes.keys():
            neigh = self._neighbor_closed(i)
            closed_neigh[i] = neigh
            num[i] = sum(red_exp[j] for j in neigh)
            den[i] = sum(red_exp[j] + black_exp[j] for j in neigh)

        N = len(self.nodes)
        grad = {k: 0.0 for k in self.nodes.keys()}

        for i in self.nodes.keys():
            den_i = den[i]
            if den_i <= 0:
                continue

            coeff = -(num[i] / (den_i * den_i)) / N
            for k in closed_neigh[i]:
                grad[k] += coeff * (1.0 - p[k])

        return grad

    def _golden_section_search(self, p, x, v, iters=25):
        """
        Minimize f((1-a)x + a v) over a in [0,1].
        """
        phi = (1 + 5**0.5) / 2
        lo, hi = 0.0, 1.0

        def mix(a):
            return {i: (1 - a) * x[i] + a * v[i] for i in x.keys()}

        c = hi - (hi - lo) / phi
        d = lo + (hi - lo) / phi
        fc = self._objective_expected_Sbar(p, mix(c))
        fd = self._objective_expected_Sbar(p, mix(d))

        for _ in range(iters):
            if fc < fd:
                hi = d
                d = c
                fd = fc
                c = hi - (hi - lo) / phi
                fc = self._objective_expected_Sbar(p, mix(c))
            else:
                lo = c
                c = d
                fc = fd
                d = lo + (hi - lo) / phi
                fd = self._objective_expected_Sbar(p, mix(d))

        return 0.5 * (lo + hi)

    def compute_curing_gradient_simplex(self, max_iters=10):
        """
        Compute x on simplex:
            x_i >= 0, sum_i x_i = B
        using a Frank-Wolfe style update.
        """
        if self.budget_B is None:
            self.budget_B = float(sum(node.delta_red for node in self.nodes.values()))

        B = self.budget_B
        node_ids = list(self.nodes.keys())
        p = self._compute_p()

        # start from uniform allocation
        x = {i: B / len(node_ids) for i in node_ids}

        for _ in range(max_iters):
            grad = self._gradient_expected_Sbar(p, x)

            # steepest descent vertex
            i_star = min(grad, key=lambda k: grad[k])

            v = {i: 0.0 for i in node_ids}
            v[i_star] = B

            alpha = self._golden_section_search(p, x, v)
            x = {i: (1 - alpha) * x[i] + alpha * v[i] for i in node_ids}

        return x

    # ------------------------------------------------------
    # Simulation
    # ------------------------------------------------------
    def get_network_metrics(self):
        U_bar = np.mean([n.get_proportion() for n in self.nodes.values()])
        S_bar = np.mean([self.get_super_urn_proportion(i) for i in self.nodes.keys()])

        wasted_budget = 0.0
        for nid in self.graph.nodes():
            if len(self.nodes[nid].draw_queue) == 0:
                continue
            draw_result, _, delta_black_step = self.nodes[nid].draw_queue[-1]
            wasted_budget += delta_black_step * draw_result

        return U_bar, S_bar, wasted_budget

    def simulate_step(self, curing_strategy="Gradient"):
        node_ids = list(self.nodes.keys())
        N = len(node_ids)
        if N == 0:
            return {}

        # default total budget = sum of delta_red over all nodes
        if self.budget_B is None:
            self.budget_B = float(sum(self.nodes[i].delta_red for i in node_ids))

        if curing_strategy == "none":
            x = {i: 0.0 for i in node_ids}
        elif curing_strategy == "Uniform":
            x = {i: self.budget_B / N for i in node_ids}
        elif curing_strategy == "Gradient":
            x = self.compute_curing_gradient_simplex(max_iters=10)
        else:
            raise ValueError(f"Unknown curing strategy: {curing_strategy}")

        proportions = {i: self.get_super_urn_proportion(i) for i in node_ids}
        self.cached_super_urn = proportions

        draws = {i: 1 if np.random.random() < proportions[i] else 0 for i in node_ids}

        for i, d in draws.items():
            self.nodes[i].update(d, delta_black_step=x[i])

        return x


# ==========================================================
# Simulation runner
# ==========================================================
class SimulationRunner:
    def __init__(
        self,
        num_steps,
        iterations,
        initial_conditions,
        initial_nodes,
        m,
        curing_type,
    ):
        self.num_steps = num_steps
        self.iterations = iterations
        self.initial_conditions = initial_conditions
        self.initial_nodes = initial_nodes
        self.m = m
        self.curing_type = curing_type

    def run_simulation(self):
        simulation_data = np.zeros((self.iterations, self.num_steps, 4))
        curing_costs = np.zeros((self.iterations, self.num_steps))

        for i in range(self.iterations):
            seed = 123 + i
            random.seed(seed)
            np.random.seed(seed)

            network = BANetwork()
            network.initialize_barabasi_albert(
                n=self.initial_nodes,
                m=self.m,
                initial_conditions=self.initial_conditions,
            )

            budget = len(network.nodes) * self.initial_conditions[1]
            network.set_budget(budget)

            for step in range(self.num_steps):
                x = network.simulate_step(curing_strategy=self.curing_type)
                curing_costs[i, step] = np.sum(list(x.values()))

                U_bar, S_bar, wasted_budget = network.get_network_metrics()
                simulation_data[i, step] = [U_bar, S_bar, wasted_budget, budget]

        return simulation_data, curing_costs


# ==========================================================
# Main
# ==========================================================
def main():
    parser = argparse.ArgumentParser(description="BA graph only + GD curing")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--initial_conditions", type=json.loads, default='[5,5]')
    parser.add_argument("--initial_nodes", type=int, default=100)
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--curing_types", type=str, default="Uniform,Gradient")
    args = parser.parse_args()

    curing_types = [s.strip() for s in args.curing_types.split(",") if s.strip()]
    results = {}
    curing_costs = {}

    for curing in curing_types:
        sim = SimulationRunner(
            num_steps=args.steps,
            iterations=args.iterations,
            initial_conditions=args.initial_conditions,
            initial_nodes=args.initial_nodes,
            m=args.m,
            curing_type=curing,
        )

        simulation_data, cost_data = sim.run_simulation()
        results[curing] = simulation_data
        curing_costs[curing] = cost_data

    # Plot exposure
    plt.figure(figsize=(10, 5))
    for curing, data in results.items():
        avg_S = np.mean(data[:, :, 1], axis=0)
        plt.plot(np.arange(1, len(avg_S) + 1), avg_S, label=curing, linewidth=2)

    plt.xlabel("Time Step")
    plt.ylabel("Network Exposure")
    plt.title("BA Graph Only: Exposure vs Time")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Save CSVs
    for curing, data in results.items():
        avg_S = np.mean(data[:, :, 1], axis=0)
        avg_W = np.mean(data[:, :, 2], axis=0)

        df = pd.DataFrame({
            "S_bar": avg_S,
            "W": avg_W,
        })
        df.to_csv(f"{curing}_ba_only_results.csv", index=False)


if __name__ == "__main__":
    main()