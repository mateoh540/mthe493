import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import time
from collections import deque
import argparse
import pandas as pd
import datetime


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
    def initialize_barabasi_albert(self, n, m):
        self.graph = nx.barabasi_albert_graph(n, m)
        for nid in self.graph.nodes():
            self.nodes[nid] = Node(nid)

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
                    delta_red=10, 
                    delta_black=5
                )
                for existing in list(self.graph.nodes()):
                    if existing != new_id:
                        self.graph.add_edge(new_id, existing)

                print(f"Added dominating (red) node {new_id}")

            else:
                # Isolated node – dominantly black, no edges
                self.graph.add_node(new_id)
                self.nodes[new_id] = Node(
                    node_id=new_id,
                    initial_red=5,
                    initial_black=10,
                    delta_red=5, 
                    delta_black=10
                )
                print(f"Added isolated (black) node {new_id}")

                # # Add one black ball to every *existing* node (global truth effect)
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

    def run_simulation(self, visualize, switch_network, num_steps, initial_nodes):
        network = Network()
        m = min(3, initial_nodes - 1)  # avoid invalid BA parameter
        network.initialize_barabasi_albert(n=initial_nodes, m=m)       
        simulation_data = np.zeros((num_steps, 2))

        if visualize:
            self.setup_real_time_visualization(network, num_steps)

        for step in range(num_steps):
            network.simulate_step()
            U_bar, S_bar = network.get_network_metrics()
            simulation_data[step] = [U_bar, S_bar]

            if switch_network:
                network.switch_network(new_nodes=1)
                if visualize:
                     self.pos = nx.spring_layout(network.graph, seed=42)

            if visualize:
                self.update_real_time_visualization(network, simulation_data, step)

        if visualize:
            plt.ioff()
            plt.show()

        # if not visualize:
        #     steps = np.arange(num_steps)
        #     plt.figure(figsize=(6, 4))
        #     plt.plot(steps, simulation_data[:, 0], 'b-', label='Ūₙ')
        #     plt.plot(steps, simulation_data[:, 1], 'g-', label='S̄ₙ')
        #     plt.xlabel('Steps')
        #     plt.ylabel('Proportion')
        #     plt.legend()
        #     plt.title('Final Results (Static Plot)')
        #     plt.grid(True, alpha=0.3)
        #     plt.show()

        return simulation_data

    # ---------- visualization helpers ----------
    def setup_real_time_visualization(self, network, num_steps):
        plt.ion()
        self.fig, (self.ax_net, self.ax_met) = plt.subplots(1, 2, figsize=(14, 6))
        self.pos = nx.spring_layout(network.graph, seed=42)
        self._draw_network(network)
        self.metrics_line_u, = self.ax_met.plot([], [], 'b-', label='Ūₙ')
        self.metrics_line_s, = self.ax_met.plot([], [], 'g-', label='S̄ₙ')
        self.ax_met.set_xlim(0, num_steps)
        self.ax_met.set_xticks(range(0, num_steps + 1, max(1, num_steps // 10)))
        self.ax_met.set_ylim(0, 1)
        self.ax_met.set_xlabel("Steps")
        self.ax_met.set_ylabel("Proportion")
        self.ax_met.legend()
        self.ax_met.grid(True, alpha=0.3)
        plt.tight_layout()

    def _draw_network(self, network):
        node_colors = [n.get_proportion() for n in network.nodes.values()]
        self.ax_net.clear()
        nx.draw_networkx(network.graph, pos=self.pos, ax=self.ax_net,
                         node_color=node_colors, cmap='coolwarm',
                         node_size=250, vmin=0, vmax=1, with_labels=True)
        self.ax_net.set_title("Network State")
        self.ax_net.axis('off')

    def update_real_time_visualization(self, network, data, step):
        self._draw_network(network)
        steps = np.arange(step + 1)
        self.metrics_line_u.set_data(steps, data[:step + 1, 0])
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
    parser.add_argument("--switch_network", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=1)
    args = parser.parse_args()

    # Run Simulation
    total_data = []
    for i in range(args.iterations):
        sim = SimulationRunner()
        data = sim.run_simulation(
            visualize=bool(args.visualize),
            switch_network = args.switch_network,
            num_steps = args.steps,
            initial_nodes= 100 
        )
        print("Final metrics (Ūₙ, S̄ₙ):", data[-1])
        total_data.append(data)

    all_data_array = np.array(total_data)  
    print(all_data_array)
    # Plot
    # Calculate and plot averages
    avg_U = np.mean(all_data_array[:, :, 0], axis=0)
    avg_S = np.mean(all_data_array[:, :, 1], axis=0)
    plt.figure(figsize=(10, 6))
    plt.plot(avg_S, 'g-', label='S̄ₙ', linewidth=2)
    plt.xlabel('Time Steps')
    plt.ylabel('Proportion')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Export all data
    iterations, steps, metrics = all_data_array.shape
    all_df = pd.DataFrame({
        'iteration': np.repeat(range(iterations), steps),
        'step': np.tile(range(steps), iterations),
        'U_bar': all_data_array[:, :, 0].flatten(),
        'S_bar': all_data_array[:, :, 1].flatten()
    })
    all_df.to_csv('simulation_all_data.csv', index=False)
    

    


if __name__ == "__main__":
    main()
