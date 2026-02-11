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

NEWS_W_OUT = 2
NEWS_W_IN  = 0
INFLUENCER_W_OUT = 2
USER_W_OUT   = 1

# ==========================================================
# Node class
# ==========================================================
class Node:
    def __init__(self, node_id, node_type, initial_red=5, initial_black=5,
                 delta_red=10, delta_black=5, memory_size=10):
        self.node_id = node_id
        self.node_type = node_type
        self.urn_red = initial_red
        self.urn_black = initial_black
        self.delta_red = delta_red
        self.delta_black = delta_black
        self.memory_size = memory_size
        self.draw_queue = deque()

    def get_proportion(self):
        """Return the proportion of red balls."""
        return self.urn_red / (self.urn_red + self.urn_black)

    def update(self, draw_result):
        """Update urn content using finite memory."""
        self.draw_queue.append(draw_result)
        if len(self.draw_queue) > self.memory_size:
            old = self.draw_queue.popleft()
        else:
            old = None

        if draw_result == 1:
            self.urn_red += self.delta_red
        else:
            self.urn_black += self.delta_black

        if old is not None:
            if old == 1:
                self.urn_red -= self.delta_red
            else:
                self.urn_black -= self.delta_black

        self.urn_red = max(0, self.urn_red)
        self.urn_black = max(0, self.urn_black)

# ==========================================================
# Network class (static + switched)
# ==========================================================
class Network:
    def __init__(self):
        self.nodes = {}
        self.graph = nx.DiGraph()
        self.p_dominating = 0.7  # probability of dominating node

    # ---------- Initializers ----------
    def initialize_barabasi_albert(self, n, m, initial_conditions):
        ba_graph = nx.barabasi_albert_graph(n, m)
        
        centrality = nx.degree_centrality(ba_graph)
        top_3 = sorted(centrality, key=centrality.get, reverse=True)[:3]
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
                        delta_red=7, 
                        delta_black=5
                    )
                else: # Black Biased
                    self.nodes[new_id] = Node(
                    node_id=new_id,
                    node_type='influencer',
                    initial_red=5,
                    initial_black=10,
                    delta_red=5, 
                    delta_black=7
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
            weight = self.graph[node_id][nb]['weight']
            neighbor = self.nodes[nb]
            total_red += neighbor.urn_red * weight
            total_balls += (neighbor.urn_red + neighbor.urn_black) * weight
        return total_red / total_balls

    def get_network_metrics(self):
        U_bar = np.mean([n.get_proportion() for n in self.nodes.values()])
        S_bar = np.mean(list(self.cached_super_urn.values()))
        return U_bar, S_bar

    # ---------- Simulation step ----------
    def simulate_step(self):
        self.cached_super_urn = {i: self.get_super_urn_proportion(i)
                                 for i in self.nodes.keys()}
        draws = {i: 1 if np.random.random() < p else 0
                 for i, p in self.cached_super_urn.items()}
        for i, d in draws.items():
            self.nodes[i].update(d)


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
    def __init__(self):
        pass

    def run_simulation(self, visualize, num_steps, iterations, initial_conditions, initial_nodes, curing_type):
        redraw_every = 10
        simulation_data = np.zeros((iterations, num_steps, 2))
        for i in range(iterations):
            network = Network()
            m = 1
            network.initialize_barabasi_albert(n=initial_nodes, m=m, initial_conditions=initial_conditions)       
            
            if visualize:
                self.setup_real_time_visualization(network, num_steps, initial_conditions)
                try:
                    for step in range(num_steps):
                        if not plt.fignum_exists(self.fig.number): #Checks if user closed the figure
                            print("Figure closed by user, stopping simulation early.")
                            break

                        network.simulate_step()
                        U_bar, S_bar = network.get_network_metrics()
                        simulation_data[i, step] = [U_bar, S_bar]
                        if step % redraw_every == 0 or step == num_steps - 1:
                            self.update_real_time_visualization(network, simulation_data[i], step, initial_conditions)

                finally: #Closes figure
                    plt.ioff()
                    plt.close('all')
            else:
                for step in range(num_steps):
                    network.simulate_step()
                    network.apply_curing(curing_type)
                    U_bar, S_bar = network.get_network_metrics()
                    simulation_data[i, step] = [U_bar, S_bar]

        return simulation_data
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
            curing_type='gradient'
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
