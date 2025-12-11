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
        self.graph = None
        self.p_dominating = 0.5  # probability of dominating node

    # ---------- Initializers ----------
    def initialize_barabasi_albert(self, n, m, initial_conditions):
        self.graph = nx.barabasi_albert_graph(n, m)
        delta_red, delta_black = initial_conditions
        for nid in self.graph.nodes():
            self.nodes[nid] = Node(nid, delta_red=delta_red, delta_black=delta_black)

    # ---------- Switching dynamics ----------
    def switch_network(self, new_nodes=1):
        start_idx = len(self.nodes)
        for i in range(new_nodes):
            new_id = start_idx + i
            is_dominating = random.random() < self.p_dominating

            if is_dominating:
                # Dominating node – high red, connected to all
                self.graph.add_node(new_id)
                self.nodes[new_id] = Node(
                    node_id=new_id,
                    initial_red=10,
                    initial_black=5,
                    delta_red=5, 
                    delta_black=5
                )
                # Connect to 70%
                for existing in list(self.graph.nodes()):
                    if existing != new_id:
                        if random.random() < 0.5:
                            self.graph.add_edge(new_id, existing)


            else:
                # Isolated node – dominantly black, no edges
                self.graph.add_node(new_id)
                self.nodes[new_id] = Node(
                    node_id=new_id,
                    initial_red=5,
                    initial_black=10,
                    delta_red=5, 
                    delta_black=5
                )
                # 50% of connecting to 3 most popular nodes
                degrees = dict(self.graph.degree())
                degrees.pop(new_id, None)
                top3 = sorted(degrees, key=degrees.get, reverse=True)[:3]
                for node in top3:
                    if random.random() < 0.5:
                        self.graph.add_edge(new_id, node)

                # Add one black ball to every *existing* node (global truth effect)
                # for existing_id, node in self.nodes.items():
                #     if existing_id != new_id:
                #         node.urn_black += 1

                # print("Added one black ball to every other node (global correction)")

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
    def simulate_step(self):
        draws = {}
        proportions = {i: self.get_super_urn_proportion(i)
                       for i in self.nodes.keys()}
        for i, p in proportions.items():
            draws[i] = 1 if np.random.random() < p else 0
        for i, d in draws.items():
            self.nodes[i].update(d)


# ==========================================================
# Simulation Runner with visualization
# ==========================================================
class SimulationRunner:
    def __init__(self):
        pass

    def run_simulation(self, visualize, switch_network, num_steps, iterations, initial_conditions, initial_nodes):

        simulation_data = np.zeros((iterations, num_steps, 2))
        for i in range(iterations):
            network = Network()
            m = 1
            network.initialize_barabasi_albert(n=initial_nodes, m=m, initial_conditions=initial_conditions)       
            
            if visualize:
                self.setup_real_time_visualization(network, num_steps, initial_conditions)
                for step in range(num_steps):
                    network.simulate_step()
                    U_bar, S_bar = network.get_network_metrics()
                    simulation_data[i, step] = [U_bar, S_bar]
                    if switch_network:
                        network.switch_network(new_nodes=1)
                        self.pos = nx.spring_layout(network.graph, seed=42)
                    self.update_real_time_visualization(network, simulation_data[i], step, initial_conditions)

                plt.ioff()
                plt.show()
                plt.close(self.fig)
            else:
                for step in range(num_steps):
                    network.simulate_step()
                    U_bar, S_bar = network.get_network_metrics()
                    simulation_data[i, step] = [U_bar, S_bar]
                    if switch_network:
                        network.switch_network(new_nodes=1)

        return simulation_data

    # ---------- visualization helpers ----------
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
        node_colors = [
            'red' if n.get_proportion() >= 0.5 else 'black'
            for n in network.nodes.values()
        ]

        keep_limits = hasattr(self, "xlim") and hasattr(self, "ylim")
        if keep_limits:
            xlim, ylim = self.xlim, self.ylim

        self.ax_net.clear()
        nx.draw_networkx(
            network.graph, pos=self.pos, ax=self.ax_net,
            node_color=node_colors,
            node_size=250, with_labels=True
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



# ==========================================================
# Entry Point
# ==========================================================
def main():
    # Arguments
    parser = argparse.ArgumentParser(description="Run Polya Switch-Network Simulation")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--visualize", type=int, default=1)
    parser.add_argument("--switch_network", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--initial_conditions", type=json.loads, default='[[5,5]]')
    args = parser.parse_args()

    # Run Simulation
    total_simulation_data = []
    for i in range(len(args.initial_conditions)):
        sim = SimulationRunner()
        simulation_data = sim.run_simulation(
            visualize=bool(args.visualize),
            switch_network = args.switch_network,
            num_steps = args.steps,
            iterations = args.iterations,
            initial_conditions = args.initial_conditions[i],
            initial_nodes= 100 
        )
        total_simulation_data.append(simulation_data)

    data = np.array(total_simulation_data)  
    print(data)
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
    initial_conditions, iterations, steps, metric = data.shape
    all_df = pd.DataFrame({
        'initial_condition': np.repeat(range(initial_conditions), steps),
        'iteration': np.repeat(range(iterations), steps),
        'step': np.tile(range(steps), iterations),
        'S_bar': data[:, :, 1].flatten()
    })
    all_df.to_csv('simulation_all_data.csv', index=False)


if __name__ == "__main__":
    main()
