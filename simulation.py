import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import time
from collections import deque
import argparse
import pandas as pd
import datetime
import json

from matplotlib.colors import LinearSegmentedColormap

RED_BLACK_CMAP = LinearSegmentedColormap.from_list("red_black", ["black", "red"])



# ==========================================================
# Node class
# ==========================================================
class Node:
    def __init__(self, node_id, initial_red=5, initial_black=5,
                 delta_red=10, delta_black=5, memory_size=10):
        self.node_id = node_id
        self.urn_red = initial_red
        self.urn_black = initial_black
        self.delta_red = float(delta_red)
        self.delta_black = float(delta_black)  # default / fallback
        self.memory_size = memory_size

        # Store (draw, delta_red_used, delta_black_used) for finite-memory undo
        self.draw_queue = deque()

    def get_proportion(self):
        total = self.urn_red + self.urn_black
        return 0.0 if total <= 0 else self.urn_red / total

    def update(self, draw_result: int, delta_black_step: float | None = None):
        """
        Update urn content using finite memory.
        draw_result: 1 (red) or 0 (black)
        delta_black_step: curing amount used *this step* if draw_result == 0
        """
        if delta_black_step is None:
            delta_black_step = self.delta_black
        delta_black_step = float(max(0.0, delta_black_step))

        # Push new draw with the actual deltas used
        self.draw_queue.append((draw_result, self.delta_red, delta_black_step))

        # Pop old if exceeding memory
        old = self.draw_queue.popleft() if len(self.draw_queue) > self.memory_size else None

        # Apply current step reinforcement
        if draw_result == 1:
            self.urn_red += self.delta_red
        else:
            self.urn_black += delta_black_step

        # Undo the popped old reinforcement
        if old is not None:
            old_draw, old_dr, old_db = old
            if old_draw == 1:
                self.urn_red -= old_dr
            else:
                self.urn_black -= old_db

        self.urn_red = max(0.0, self.urn_red)
        self.urn_black = max(0.0, self.urn_black)

    

# ==========================================================
# Network class (static + switched)
# ==========================================================
class Network:
    def __init__(self):
        self.nodes = {}
        self.graph = None
        self.p_dominating = 0.7  # probability of dominating node

        # Budget for curing per step (set later)
        self.budget_B = None

    def set_budget(self, B: float):
        self.budget_B = float(B)

    def _neighbor_closed(self, i):
        """Closed neighborhood N_i^+ = {i} U neighbors(i)."""
        return [i] + list(self.graph.neighbors(i))

    def _compute_p(self):
        """Current p_i = S_{i,n-1} for all nodes."""
        return {i: self.get_super_urn_proportion(i) for i in self.nodes.keys()}

    def _expected_red_black_after_step(self, p, x):
        """
        Compute expected individual urn contents AFTER this step,
        using draw probabilities p_i and curing allocation x_i.

        Returns:
          red_exp[i], black_exp[i]
        """
        red_exp = {}
        black_exp = {}

        for i, node in self.nodes.items():
            # Determine whether a removal will happen (finite memory)
            will_pop = (len(node.draw_queue) >= node.memory_size)
            popped = node.draw_queue[0] if will_pop else None

            # Expected add (current step)
            add_red = node.delta_red * p[i]
            add_black = x[i] * (1.0 - p[i])

            # Deterministic remove (because popped draw is known)
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
        f(x) = expected network exposure after the step, approximated by
        plugging in expected urn contents.
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
        Analytic gradient of the approximation.

        Each super-urn term looks like num_i / den_i where den_i depends linearly
        on x_j through (1-p_j)*x_j for j in N_i^+.

        df/dx_k = -(1/N) * sum_{i: k in N_i^+} num_i * (1-p_k) / den_i^2
        """
        red_exp, black_exp = self._expected_red_black_after_step(p, x)

        # Precompute num_i and den_i for each node i
        num = {}
        den = {}
        closed_neigh = {}
        for i in self.nodes.keys():
            neigh = self._neighbor_closed(i)
            closed_neigh[i] = neigh
            num_i = sum(red_exp[j] for j in neigh)
            den_i = sum(red_exp[j] + black_exp[j] for j in neigh)
            num[i] = num_i
            den[i] = den_i

        N = len(self.nodes)
        grad = {k: 0.0 for k in self.nodes.keys()}

        # For each k, accumulate contributions from i where k in N_i^+
        # Efficient way: loop i then update all k in its neighborhood
        for i in self.nodes.keys():
            den_i = den[i]
            if den_i <= 0:
                continue
            coeff = - (num[i] / (den_i * den_i)) / N  # common factor
            for k in closed_neigh[i]:
                grad[k] += coeff * (1.0 - p[k])

        return grad

    def _golden_section_search(self, p, x, v, iters=25):
        """
        Minimize f((1-a)x + a v) over a in [0,1] via golden-section search.
        """
        phi = (1 + 5 ** 0.5) / 2
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
        Compute curing allocation x (Delta_b per node) on simplex sum x = B, x>=0
        using simplex-constrained gradient descent (Algorithm-2 style).
        """
        if self.budget_B is None:
            # reasonable default: same order as total delta_red
            self.budget_B = float(sum(n.delta_red for n in self.nodes.values()))

        B = self.budget_B
        node_ids = list(self.nodes.keys())

        # Current infection probabilities p_i = S_{i,n-1}
        p = self._compute_p()

        # Start from uniform allocation
        x = {i: B / len(node_ids) for i in node_ids}

        for _ in range(max_iters):
            grad = self._gradient_expected_Sbar(p, x)

            # Pick coordinate with most negative partial (steepest descent)
            i_star = min(grad, key=lambda k: grad[k])

            # Vertex allocation: all budget on i_star
            v = {i: 0.0 for i in node_ids}
            v[i_star] = B

            # Line search alpha in [0,1]
            alpha = self._golden_section_search(p, x, v)

            # Update
            x = {i: (1 - alpha) * x[i] + alpha * v[i] for i in node_ids}

        return x


    # ---------- Initializers ----------
    def initialize_barabasi_albert(self, n, m, initial_conditions):
        self.graph = nx.barabasi_albert_graph(n, m)
        delta_red, delta_black = initial_conditions
        for nid in self.graph.nodes():
            self.nodes[nid] = Node(nid, delta_red=delta_red, delta_black=delta_black)

        self.switch_network(10)
        

    # ---------- Switching dynamics ----------
    def switch_network(self, new_nodes):
        start_idx = len(self.nodes)
        dominating_balls = 1
        isolated_balls = 10
        for i in range(new_nodes):
            new_id = start_idx + i
            # === Polya Process ===
            if random.random() < (dominating_balls / (dominating_balls + isolated_balls)):
                dominating_balls = dominating_balls + 1
                node_type = 'dominating'
            else:
                isolated_balls = isolated_balls + 1
                node_type = 'isolated'
            

            existing_nodes = [n for n in self.graph.nodes()]
            self.graph.add_node(new_id)
 
            if node_type == 'dominating': #
                #Red Biased 
                if random.random() < 0.5:
                    self.nodes[new_id] = Node(
                        node_id=new_id,
                        initial_red=10,
                        initial_black=5,
                        delta_red=7, 
                        delta_black=5
                    )
                else: # Black Biased
                    self.nodes[new_id] = Node(
                    node_id=new_id,
                    initial_red=5,
                    initial_black=10,
                    delta_red=5, 
                    delta_black=7
                )  
                m = int(np.ceil(0.4*len(existing_nodes))) # Connect to 40% of nodes
            else:# Neutral User

                self.nodes[new_id] = Node(
                    node_id=new_id,
                    initial_red=5,
                    initial_black=5,
                    delta_red=5, 
                    delta_black=5
                )
                m = int(np.ceil(0.1*len(existing_nodes))) # Connect to 10% of Nodes
        
            # Choose nodes to connect to (preferential attachment)
            chosen = set()
            for _ in range(m):
                candidates = [n for n in existing_nodes if n not in chosen]
                degrees = {n: self.graph.degree(n) for n in candidates}
                total_deg = sum(degrees.values())

                r = random.random() * total_deg
                acc = 0.0
                pick = candidates[-1]
                for n in candidates:
                    acc += degrees[n]
                    if acc >= r:
                        pick = n
                        break
            
                chosen.add(pick)
                self.graph.add_edge(new_id, pick)

    # ---------- Super-urn calculations ----------
    def get_super_urn_proportion(self, node_id):
        node = self.nodes[node_id]
        neighbors = list(self.graph.neighbors(node_id))
        total_red = node.urn_red
        total_balls = node.urn_red + node.urn_black
        for nb in neighbors:
            neighbor = self.nodes[nb]
            total_red += neighbor.urn_red
            total_balls += neighbor.urn_red + neighbor.urn_black
        return total_red / total_balls

    def get_network_metrics(self):
        U_bar = np.mean([n.get_proportion() for n in self.nodes.values()])
        S_bar = np.mean([self.get_super_urn_proportion(i)
                         for i in self.nodes.keys()])
        return U_bar, S_bar

    # ---------- Simulation step ----------
    def simulate_step(self, curing_strategy: str = "none"):
        """
        One time-step of the network Polya process with optional curing.

        curing_strategy:
        - "none":    no extra curing this step (delta_black_step = 0 for everyone)
        - "uniform": spread the budget B evenly across all nodes (sum x_i = B)
        - "gradient":choose x on the simplex (sum x_i = B) to reduce expected next exposure

        IMPORTANT:
        - Step 1 decides x_i (curing amounts) for THIS step.
        - Step 2 draws using CURRENT super-urn proportions (curing doesn't affect draws until next step).
        - Step 3 updates urns using x_i only when a node draws black.
        """
        node_ids = list(self.nodes.keys())
        N = len(node_ids)
        if N == 0:
            return

        # ----------------------------
        # 1) Choose curing allocation x
        # ----------------------------
        # If strategy needs a budget, ensure it exists.
        if curing_strategy in ("uniform", "gradient"):
            if self.budget_B is None:
                raise ValueError(
                    "budget_B is not set. Call network.set_budget(B) after initialization."
                )
            B = float(self.budget_B)

        if curing_strategy == "none":
            # Truly no curing this step
            x = {i: 0.0 for i in node_ids}

        elif curing_strategy == "uniform":
            # Budget-feasible baseline: spread B evenly
            x = {i: B / N for i in node_ids}

        elif curing_strategy == "gradient":
            # Budget-feasible optimization on simplex
            x = self.compute_curing_gradient_simplex(max_iters=10)

        else:
            raise ValueError(
                f"Unknown curing_strategy='{curing_strategy}'. "
                "Use 'none', 'uniform', or 'gradient'."
            )

        # ----------------------------
        # 2) Draw from current super-urns
        # ----------------------------
        proportions = {i: self.get_super_urn_proportion(i) for i in node_ids}
        draws = {i: 1 if np.random.random() < proportions[i] else 0 for i in node_ids}

        # ----------------------------
        # 3) Update urns using this step's curing x_i (only if draw is black)
        # ----------------------------
        for i, d in draws.items():
            self.nodes[i].update(d, delta_black_step=x[i])




    def apply_curing(self, curing_strategy):
        match curing_strategy:
            case "gradient":
                self.nodes
                self.graph
                self.get_super_urn_proportion(1)
            case "heuristic":
                self.nodes
                self.graph
                self.get_super_urn_proportion(1)
            case "supermartingale":
                self.nodes
                self.graph
                self.get_super_urn_proportion(1)

# ==========================================================
# Simulation Runner with visualization
# ==========================================================
class SimulationRunner:
    def __init__(self):
        pass

    def run_simulation(self, visualize, num_steps, iterations, initial_conditions, initial_nodes, curing_type):
    
        simulation_data = np.zeros((iterations, num_steps, 2))
        for i in range(iterations):
            network = Network()
            m = 1
            network.initialize_barabasi_albert(n=initial_nodes, m=m, initial_conditions=initial_conditions)       
            network.set_budget(len(network.nodes) * initial_conditions[1])



            if visualize:
                self.setup_real_time_visualization(network, num_steps, initial_conditions)
                for step in range(num_steps):
                    network.simulate_step(curing_strategy=curing_type)
                    U_bar, S_bar = network.get_network_metrics()
                    simulation_data[i, step] = [U_bar, S_bar]
                    self.update_real_time_visualization(network, simulation_data[i], step, initial_conditions)

                plt.ioff()
                plt.show()
                plt.close(self.fig)
            else:
                for step in range(num_steps):
                    network.simulate_step(curing_strategy=curing_type)
                    U_bar, S_bar = network.get_network_metrics()
                    simulation_data[i, step] = [U_bar, S_bar]

        return simulation_data
    def setup_real_time_visualization(self, network, num_steps, initial_conditions):
        plt.ion()
        self.fig, (self.ax_net, self.ax_met) = plt.subplots(
            1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [2, 1]}
        )

        # Initial layout computed ONCE
        self.pos = nx.spring_layout(network.graph, seed=42)

        # Draw once to fix initial limits
        self._draw_network(network, initial_conditions)

        # Save initial axis limits so they stay fixed later
        self.xlim = self.ax_net.get_xlim()
        self.ylim = self.ax_net.get_ylim()

        self.metrics_line_s, = self.ax_met.plot([], [], 'g-', label=r'Network Exposure ($\bar{S}_n$)')
        self.ax_met.set_xlim(0, num_steps)
        self.ax_met.set_xticks(range(0, num_steps + 1, max(1, num_steps // 10)))
        self.ax_met.set_ylim(0, 1)
        self.ax_met.set_xlabel("Steps")
        self.ax_met.set_ylabel("Proportion")
        self.ax_met.legend()
        self.ax_met.grid(True, alpha=0.3)
        plt.tight_layout()

    def _update_positions_for_new_nodes(self, network):
        """
        Ensure every node in network.graph has a position in self.pos.
        New nodes get placed near their neighbors or near the existing center.
        """
        existing_nodes = set(self.pos.keys())

        # Precompute center of existing layout
        xs = [p[0] for p in self.pos.values()]
        ys = [p[1] for p in self.pos.values()]
        cx, cy = float(np.mean(xs)), float(np.mean(ys))

        for node in network.graph.nodes():
            if node in existing_nodes:
                continue

            neighbors = [n for n in network.graph.neighbors(node) if n in self.pos]

            if neighbors:
                # Place at the mean of neighbor positions
                nxs = [self.pos[n][0] for n in neighbors]
                nys = [self.pos[n][1] for n in neighbors]
                self.pos[node] = (float(np.mean(nxs)), float(np.mean(nys)))
            else:
                # No neighbors with known positions yet: drop near center with small jitter
                self.pos[node] = (
                    cx + 0.1 * np.random.randn(),
                    cy + 0.1 * np.random.randn(),
                )

    def _draw_network(self, network, initial_conditions):
        # Map proportion to hard colors

        keep_limits = hasattr(self, "xlim") and hasattr(self, "ylim")
        if keep_limits:
            xlim, ylim = self.xlim, self.ylim
        node_colors = self._get_node_colors(network)
        self.ax_net.clear()
        nx.draw_networkx(
            network.graph, pos=self.pos, ax=self.ax_net,
            node_color=node_colors, node_size=200,
            cmap='coolwarm', vmin=0, vmax=1
        )
        self.ax_net.set_title("Network")
        self.ax_net.axis('off')

        # Initial Conditions Box (unchanged)
        textstr = (
            rf'$\Delta_r = {initial_conditions[0]},\ \Delta_b = {initial_conditions[1]}$'
            '\n'
            r'$r_{(i, n)} = 5,\ b_{(i,n)} = 5$'
        )
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        self.ax_net.text(
            0.05, 0.95, textstr,
            transform=self.ax_net.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=props
        )

        if keep_limits:
            self.ax_net.set_xlim(xlim)
            self.ax_net.set_ylim(ylim)
        else:
            self.xlim = self.ax_net.get_xlim()
            self.ylim = self.ax_net.get_ylim()

    def update_real_time_visualization(self, network, data, step, initial_conditions):
        # Give positions to any new nodes without disturbing old ones
        self._update_positions_for_new_nodes(network)

        # Redraw network with fixed layout & fixed axis limits
        self._draw_network(network, initial_conditions)

        # Metrics panel stays dynamic
        steps = np.arange(step + 1)
        self.metrics_line_s.set_data(steps, data[:step + 1, 1])
        self.ax_met.relim()
        self.ax_met.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _get_node_colors(self, network):
        return [network.nodes[node_id].get_proportion() 
                for node_id in network.graph.nodes()]

# ==========================================================
# Entry Point
# ==========================================================
def main():
    # Arguments
    parser = argparse.ArgumentParser(description="Run Polya Network Simulation")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--visualize", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--initial_conditions", type=json.loads, default='[[5,5]]')
    parser.add_argument("--curing_type", type=str, default='gradient')
    args = parser.parse_args()

    # === Run Simulation === 
    total_simulation_data = []
    for i in range(len(args.initial_conditions)):
        sim = SimulationRunner()
        simulation_data = sim.run_simulation(
            visualize=bool(args.visualize),
            num_steps = args.steps,
            iterations = args.iterations,
            initial_conditions = args.initial_conditions[i],
            initial_nodes= 100,
            curing_type=args.curing_type
        )
        total_simulation_data.append(simulation_data)


    # === Data ===
    data = np.array(total_simulation_data)  
    print(data)

    # === Plotting Final Results

    # Plot
    # Calculate and plot averages
    plt.figure(figsize=(10, 6))
    for i in range(len(args.initial_conditions)):
        avg_S = np.mean(data[i, :, :, 1], axis=0)
        avg_S = np.insert(avg_S, 0, 0.5)
        plt.plot(avg_S, label=f'S̄ₙ for Initial Conditions {args.initial_conditions[i]}', linewidth=2)
    plt.xlabel('Time Steps')
    plt.ylabel('Proportion')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Export all data
    # initial_conditions, iterations, steps, metric = data.shape
    # all_df = pd.DataFrame({
    #     'initial_condition': np.repeat(range(initial_conditions), steps),
    #     'iteration': np.repeat(range(iterations), steps),
    #     'step': np.tile(range(steps), iterations),
    #     'S_bar': data[:, :, 1].flatten()
    # })
    # all_df.to_csv('simulation_all_data.csv', index=False)


if __name__ == "__main__":
    main()
