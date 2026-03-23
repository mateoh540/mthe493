import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import logging
from collections import deque
import argparse
import pandas as pd
from datetime import datetime
import json
from approximate_dp import budget_plan
from evaluation_metrics import evaluate_all_strategies, summarize_results

from matplotlib.colors import LinearSegmentedColormap

RED_BLACK_CMAP = LinearSegmentedColormap.from_list("red_black", ["black", "red"])

NEWS_W_OUT = 2
NEWS_W_IN  = 0
INFLUENCER_W_OUT = 2
USER_W_OUT   = 1

# ==========================================================
# Node class
# ==========================================================
class Node:
    def __init__(self, node_id, node_type, initial_red=5, initial_black=5,
                 delta_red=5, delta_black=5, memory_size=10):
        self.node_id = node_id
        self.node_type = node_type
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
# Network class
# ==========================================================
class Network:
    def __init__(self):
        self.nodes = {}
        self.graph = nx.DiGraph()
        self.p_dominating = 0.8  
        self.cached_super_urn = {} 
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
    def initialize_barabasi_albert(self, n, m, additional_nodes, initial_conditions):
        ba_graph = nx.barabasi_albert_graph(n, m)
        self.centrality = nx.betweenness_centrality(ba_graph, weight=None, normalized=True)
        top_3 = sorted(self.centrality, key=self.centrality.get, reverse=True)[:3]
        self.news_nodes = set(top_3)

        self.graph = nx.DiGraph()
        delta_red, delta_black = initial_conditions
        for nid in ba_graph.nodes():
            if nid in self.news_nodes:
                node_type = 'news'
            else:
                node_type = 'user'
            self.graph.add_node(nid)
            self.nodes[nid] = Node(nid, node_type=node_type, delta_red=delta_red, delta_black=delta_black)

        for u, v in ba_graph.edges():
            self.graph.add_edge(u, v, weight=self.edge_weight(u, v))
            self.graph.add_edge(v, u, weight=self.edge_weight(v, u))

        self.switch_network(additional_nodes)
        self.compute_centrality_terms()
        

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
            

            existing_nodes = list(self.graph.nodes())
            self.graph.add_node(new_id)
 
            if node_type == 'dominating': #
                #Red Biased 
                if random.random() < 0.5:
                    self.nodes[new_id] = Node(
                        node_id=new_id,
                        node_type='influencer',
                        initial_red=10,
                        initial_black=5,
                        delta_red=5, 
                        delta_black=5
                    )
                else: # Black Biased
                    self.nodes[new_id] = Node(
                    node_id=new_id,
                    node_type='influencer',
                    initial_red=5,
                    initial_black=10,
                    delta_red=5, 
                    delta_black=5
                )  
                m = int(np.ceil(0.4*len(existing_nodes))) # Connect to 40% of nodes
            else:# Neutral User

                self.nodes[new_id] = Node(
                    node_id=new_id,
                    node_type='user',
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
               
                self.graph.add_edge(new_id, pick, weight=self.edge_weight(new_id, pick))
                self.graph.add_edge(pick, new_id, weight=self.edge_weight(pick, new_id))

    # ---------- Super-urn calculations ----------
    def get_super_urn_proportion(self, node_id):
        node = self.nodes[node_id]
        total_red = node.urn_red
        total_balls = node.urn_red + node.urn_black
        for nb in self.graph.predecessors(node_id):
            weight = self.graph[nb][node_id]['weight']
            neighbor = self.nodes[nb]
            total_red += neighbor.urn_red * weight
            total_balls += (neighbor.urn_red + neighbor.urn_black) * weight
        return total_red / total_balls

    def get_network_metrics(self):
        U_bar = np.mean([n.get_proportion() for n in self.nodes.values()])
        S_bar = np.mean([self.get_super_urn_proportion(i) for i in self.nodes.keys()])

        wasted_budget = 0.0
        for nid in self.graph.nodes():
            if len(self.nodes[nid].draw_queue) > 0:
                draw_result, _, delta_black_step = self.nodes[nid].draw_queue[-1]
                wasted_budget += delta_black_step * draw_result

        return U_bar, S_bar, wasted_budget
        
    
    def compute_centrality_terms(self):
        node_ids = list(self.nodes.keys())
        self.impact = {
            i: sum(data.get("weight", 1.0) for _, _, data in self.graph.out_edges(i, data=True))
            for i in node_ids
        }

        Gdist = nx.DiGraph()
        Gdist.add_nodes_from(self.graph.nodes())   # <-- important

        for u, v, data in self.graph.edges(data=True):
            w = float(data.get("weight", 1.0))
            if w > 0:
                Gdist.add_edge(u, v, weight=1.0 / w)

        self.centrality = nx.betweenness_centrality(Gdist, weight="weight", normalized=True)

    def simulate_step(self, curing_strategy: str = "none"):
        node_ids = list(self.nodes.keys())
        N = len(node_ids)
        if N == 0:
            return

        # Curing Strategy
        # Total Budget is sum of delta_red
        if self.budget_B is None:
            self.budget_B = float(sum(self.nodes[i].delta_red for i in node_ids))

        if curing_strategy == "none":
            # Truly no curing this step
            x = {i: 0.0 for i in node_ids}

        elif curing_strategy == "Uniform":
            # Budget-feasible baseline: spread B evenly
            x = {i: self.budget_B / N for i in node_ids}

        elif curing_strategy == "Gradient":
            # Budget-feasible optimization on simplex
            x = self.compute_curing_gradient_simplex(max_iters=10)

        elif curing_strategy == 'Centrality':
            S_prev = {i: self.get_super_urn_proportion(i) for i in node_ids}
            # Compute Ratios
            node_weight = {}
            for i in node_ids:
                node_weight[i] = self.impact[i] * self.centrality[i] * S_prev[i]

            total_weight = sum(node_weight.values())
            if total_weight > 0:
                x = {i: self.budget_B * node_weight[i] / total_weight for i in node_ids}
            else:
                x = {i: self.budget_B / len(node_ids) for i in node_ids}

        # Draw from Super Urn
        proportions = {i: self.get_super_urn_proportion(i) for i in node_ids}
        self.cached_super_urn = proportions
        draws = {i: 1 if np.random.random() < proportions[i] else 0 for i in node_ids}

        #Update Urns
        for i, d in draws.items():
            self.nodes[i].update(d, delta_black_step=x[i])

        return x


    def edge_weight(self, u, v):
        if self.nodes[v].node_type == 'news':
            return NEWS_W_IN

        src_type = self.nodes[u].node_type
        if src_type == 'news':
            return NEWS_W_OUT
        elif src_type == 'influencer':
            return INFLUENCER_W_OUT
        else:
            return USER_W_OUT

        

# ==========================================================
# Simulation Runner with visualization
# ==========================================================
class SimulationRunner:
    def __init__(self, visualize, num_steps, iterations, initial_conditions, initial_nodes, additional_nodes,curing_type, horizon):
        self.visualize = visualize
        self.num_steps = num_steps
        self.iterations = iterations
        self.initial_conditions = initial_conditions
        self.initial_nodes = initial_nodes
        self.additional_nodes = additional_nodes
        self.curing_type = curing_type
        self.horizon = horizon

    def run_simulation(self):
        redraw_every = 10
        simulation_data = np.zeros((self.iterations, self.num_steps, 4))
        curing_costs = np.zeros((self.iterations, self.num_steps))
        plan = []
        plan_idx = 0

        for i in range(self.iterations):
            logging.info(f"Iteration: {i}")
            # Set Seed``
            seed = 100 + i  # i = iteration
            random.seed(seed)
            np.random.seed(seed)

            # Create Network
            logging.info("Creating Network")
            network = Network()
            m = 1
            network.initialize_barabasi_albert(n=self.initial_nodes, m=m, additional_nodes = self.additional_nodes,initial_conditions=self.initial_conditions)  #Initialize the Graph

            # Run the Simulation
            if self.visualize:
                self.setup_real_time_visualization(network, self.num_steps, self.initial_conditions)

        
            # Set Budget
            # Set Budget
            default_budget = len(network.nodes) * self.initial_conditions[1]
            if self.horizon <= 1:
                network.set_budget(default_budget)
            else:
                B_total = self.horizon * default_budget
                plan = self.create_budget_plan(network, self.horizon, B_total=B_total)
                plan_idx = 0
                network.set_budget(plan[plan_idx])
                plan_idx += 1

            U_bar, S_bar, wasted_budget = network.get_network_metrics()
            actual_budget = network.budget_B if network.budget_B is not None else 0.0
            simulation_data[i, 0] = [U_bar, S_bar, wasted_budget, actual_budget]

            logging.info("Beginning Simulation")
            for step in range(self.num_steps - 1):
                if self.visualize and not plt.fignum_exists(self.fig.number): #Checks if user closed the figure
                    print("Figure closed by user, stopping simulation early.")
                    break
                
                if self.horizon > 1:
                    # Set Budget for this step
                    if (step % self.horizon == 0) or (plan_idx >= len(plan)):
                        plan_idx = 0
                    network.set_budget(plan[plan_idx])
                    plan_idx += 1 

                # Simulate Step
                x = network.simulate_step(curing_strategy=self.curing_type)
                curing_costs[i, step] = np.sum(list(x.values()))

                U_bar, S_bar, wasted_budget = network.get_network_metrics()
                actual_budget = network.budget_B if network.budget_B is not None else 0.0
                simulation_data[i, step + 1] = [U_bar, S_bar, wasted_budget, actual_budget]
               

                if self.visualize:
                    self.update_real_time_visualization(network, simulation_data[i], step, self.initial_conditions)
            logging.info("Ending Simulation")

            if self.visualize:
                plt.ioff()
                plt.close('all')
                
        return simulation_data, curing_costs
    
    def create_budget_plan(self, network, horizon, B_total):
        """
        Returns the Plan for Budget Allocation over the finite time Horizon
        """
        #budget_plan
        plan = []
        plan_idx = 0
        delta = B_total / 100
        n_rollouts = 10
        # Find the Budget Plan
        # if you have total remaining budget for the entire run:
        plan = budget_plan(network, budget=B_total, delta=delta, horizon=horizon, n_rollouts=n_rollouts, curing_strategy=self.curing_type, base_seed=123, alpha=1)
        print(plan)
        return plan
        
    def setup_real_time_visualization(self, network, num_steps, initial_conditions):
        plt.ion()
        self.fig, (self.ax_net, self.ax_met) = plt.subplots(
            1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [2, 1]}
        )

        degrees = dict(network.graph.degree())
        self.pos = nx.spring_layout(network.graph, seed=42, k=3.0, iterations=100, weight=None)
        for n in self.pos:
            scale = np.log1p(degrees[n])
            self.pos[n] = self.pos[n] * scale

        # Draw once to fix initial limits
        self._draw_network(network, initial_conditions)

        # Save initial axis limits so they stay fixed later
        self.xlim = self.ax_net.get_xlim()
        self.ylim = self.ax_net.get_ylim()

        self.metrics_line_s, = self.ax_met.plot([], [], 'g-', label=r'$\tilde{S}_n$')
        self.ax_met.set_xlim(0, num_steps)
        self.ax_met.set_xticks(range(0, num_steps + 1, max(1, num_steps // 10)))
        self.ax_met.set_ylim(0, 1)
        self.ax_met.set_xlabel("Steps")
        self.ax_met.set_ylabel(r'Network Exposure ($\tilde{S}_n$)')
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
        self._update_positions_for_new_nodes(network)
        self._draw_network(network, initial_conditions)

        steps = np.arange(step + 2)
        self.metrics_line_s.set_data(steps, data[:step + 2, 1])

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _get_node_colors(self, network):
        return [network.get_super_urn_proportion(node_id) for node_id in network.graph.nodes()]

def run_evaluation(results, curing_costs, iterations, steps, alpha=0.33):
    evaluation_results = evaluate_all_strategies(results, curing_costs, alpha=alpha)
    evaluation_df = summarize_results(evaluation_results)
    return evaluation_df

# ==========================================================
# Entry Point
# ==========================================================
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    # Arguments
    parser = argparse.ArgumentParser(description="Run Polya Network Simulation")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--visualize", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--initial_conditions", type=json.loads, default='[5,5]')
    parser.add_argument("--initial_nodes", type=int, default='1000')
    parser.add_argument("--additional_nodes", type=int, default='0')
    parser.add_argument("--curing_types",type=str, default="Uniform, Centrality")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--evaluate", type=int, default=1)
    args = parser.parse_args()

    # List of Curing Types
    curing_types = [s.strip() for s in args.curing_types.split(",") if s.strip()]

    # === Run Simulation === 
    logging.info("Starting Simulation with curing types: %s", curing_types)
    results = {} 
    curing_costs = {}
    for curing in curing_types:
        logging.info("Simulating for curing type: %s", curing)
        sim = SimulationRunner(visualize=bool(args.visualize),
            num_steps = args.steps,
            iterations = args.iterations,
            initial_conditions = args.initial_conditions,
            initial_nodes= args.initial_nodes,
            additional_nodes=args.additional_nodes,
            curing_type=curing,
            horizon=args.horizon
        )
        simulation_data, cost_data = sim.run_simulation()

        results[curing] = np.array(simulation_data)
        curing_costs[curing] = np.array(cost_data)


    # === Evaluation Metrics ====
    if args.evaluate:
        evaluation_df = run_evaluation(
            results=results,
            curing_costs=curing_costs,
            iterations=args.iterations,
            steps=args.steps,
            alpha=0.10
        )
        evaluation_df.to_csv(
        f"results/data/evaluation_metrics_initial_nodes_{args.initial_nodes}_additional_nodes_{args.additional_nodes}_horizon_{args.horizon}_iterations_{args.iterations}_steps_{args.steps}.csv",
        index=False
        )


    # === Plotting Final Results ===
    logging.info("Plotting Final Results")
    style_map = {
        "Uniform":    {"label": "(i) Uniform",    "marker": "s"},
        "Centrality": {"label": "(ii) Centrality","marker": "^"},
        "Gradient":   {"label": "(iii) Gradient",   "marker": "D"},
    }
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.dpi": 150,
    })

    # Exposure
    plt.figure(figsize=(10, 5))
    for curing, data in results.items():
        avg_S = np.mean(data[:, :, 1], axis=0)

        style = style_map.get(curing, {"label": curing, "marker": "o"})
        plt.plot(
            np.arange(1, len(avg_S) + 1),
            avg_S,
            label=style["label"],
            linewidth=1.8,
            marker=style["marker"],
            markersize=7,
            markevery=max(1, len(avg_S)//10)
        )
        
    plt.xlabel('Time Step')
    plt.ylabel(r'$ \tilde S_t$: Network Exposure')
    plt.xlim(0, args.steps)
    plt.ylim(0, 0.6)
    plt.legend(frameon=True, fancybox=False, edgecolor="black", loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"results/figures/network_exposure_initial_nodes_{args.initial_nodes}_additional_nodes_{args.additional_nodes}_horizon_{args.horizon}_iterations_{args.iterations}_steps_{args.steps}.png", dpi=200)
    plt.close()

    # Wasted Curing
    plt.figure(figsize=(10, 5))
    for curing, data in results.items():
        ratio = np.divide(
            data[:, :, 2],
            data[:, :, 3],
            out=np.zeros_like(data[:, :, 2], dtype=float),
            where=data[:, :, 3] > 0
        )
        avg_wasted_budget_ratio = np.mean(ratio, axis=0)
        style = style_map.get(curing, {"label": curing, "marker": "o"})
        if args.horizon > 1:
            avg_wasted_budget_ratio = np.where(avg_wasted_budget_ratio == 0,
                                       np.nan,
                                       avg_wasted_budget_ratio)
            plt.scatter(
                np.arange(1, len(avg_wasted_budget_ratio) + 1),
                avg_wasted_budget_ratio,
                label=style["label"],
                marker=style["marker"],
                s=40
            )
        else: 
            plt.plot(
                np.arange(1, len(avg_wasted_budget_ratio) + 1),
                avg_wasted_budget_ratio,
                label=style["label"],
                linewidth=1.8,
                marker=style["marker"],
                markersize=7,
                markevery=max(1, len(avg_S)//10)
            )
  
    plt.xlabel('Time Step')
    plt.ylabel(r'$W_t$: Wasted Budget Ratio')
    plt.xlim(2, args.steps)
    plt.legend(frameon=True, fancybox=False, edgecolor="black", loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"results/figures/wasted_budget_ratio_initial_nodes_{args.initial_nodes}_additional_nodes_{args.additional_nodes}_horizon_{args.horizon}_iterations_{args.iterations}_steps_{args.steps}.png", dpi=200)
    plt.close()

    # Save Data
    logging.info("Saving simulation data to CSV")

    for curing, data in results.items():
        avg_S = np.mean(data[:, :, 1], axis=0)
        ratio = np.divide(
            data[:, :, 2],
            data[:, :, 3],
            out=np.zeros_like(data[:, :, 2], dtype=float),
            where=data[:, :, 3] > 0
        )
        avg_wasted_budget_ratio = np.mean(ratio, axis=0)
        df = pd.DataFrame({'S_bar': avg_S, 'W': avg_wasted_budget_ratio})
        df.to_csv(f'results/data/{curing}_initial_nodes_{args.initial_nodes}_additional_nodes_{args.additional_nodes}_horizon_{args.horizon}_iterations_{args.iterations}_steps_{args.steps}.csv', index=False)
    
    logging.info(f"Simulation finished")
if __name__ == "__main__":
    main()
