# === Libraries ===
# Graphs
import networkx as nx   # For Graphs
# Plotting
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
import seaborn as sns
# Numerical Methods
import numpy as np
import scipy as sp
import math
import random
import time


# === Declarations ===
class Node:
    def __init__(self, node_id, initial_red=5, initial_black=5, delta_red=1, delta_black=1, memory_size=10):
        self.node_id = node_id
        self.urn_red = initial_red
        self.urn_black = initial_black
        self.delta_red = delta_red
        self.delta_black = delta_black
        self.memory_size = memory_size
        self.history = [] 

    def get_proportion(self): # Returns the Proportion of Red Balls in the Urn
        return self.urn_red / (self.urn_red + self.urn_black)
    
    def update(self, draw_result): # Updates the Contents of the Urn
        # Add new balls
        if draw_result == 1:
            self.urn_red += self.delta_red
        else:
            self.urn_black += self.delta_black

        # Remove old bars from memory


class Network:
    def __init__(self):
        self.nodes = {}
        pass
    def initialize_erdos_renyi(self, num_nodes, probability): # Initialize a Erdos-Renyi Graph
        self.graph = nx.erdos_renyi_graph(num_nodes, probability)
        for node_id in self.graph.nodes():
            self.nodes[node_id] = Node(
                node_id=node_id,

            )
        
    def get_super_urn_proportion(self, node_id): # Get Super Urn Properties for a Node
        # Get Node and Neighbours
        node = self.nodes[node_id]
        neighbors = list(self.graph.neighbors(node_id))

        # Get Total Red and Black Balls
        total_red = node.urn_red
        total_balls = node.urn_red + node.urn_black
        for neighbor_id in neighbors:
            if neighbor_id in self.nodes:
                neighbor = self.nodes[neighbor_id]
                total_red += neighbor.urn_red
                total_balls += neighbor.urn_red + neighbor.urn_black
        return total_red / total_balls

    def get_network_metrics(self): # Get the Contagion Metrics of the Network
        U_bar = 0
        S_bar = 0
        n = len(self.nodes)
        for node_id, node in self.nodes.items():
            U_bar += node.get_proportion()
            S_bar += self.get_super_urn_proportion(node_id)
        U_bar /= n
        S_bar /= n
        return U_bar, S_bar

    def simulate_step(self): # Simulate a Step in the Network Smulation
        draws = {}
        proportions = {}

        # Get Proportions
        for node_id in self.nodes:
            proportions[node_id] = self.get_super_urn_proportion(node_id)
        
        # Perform Draws
        for node_id, proportion in proportions.items():
            draw_result = 1 if np.random.random() < proportion else 0
            draws[node_id] = draw_result
        
        # Update all nodes
        for node_id, draw_result in draws.items():
            self.nodes[node_id].update(draw_result)
class SimulationRunner:
    def __init__(self):
        pass
    def run_simulation(self, network_type, num_steps, visualize):
        if network_type == "erdos renyi":
            network = Network()
            network.initialize_erdos_renyi(50, 0.5)
        simulation_data = np.zeros((num_steps, 2))

        if visualize:
            # Initialize real-time visualization
            self.setup_real_time_visualization(network)

        for i in range(num_steps):
            network.simulate_step()            
            U_bar, S_bar = network.get_network_metrics()
            simulation_data[i, 0] = U_bar
            simulation_data[i, 1] = S_bar
            if visualize:
                self.update_real_time_visualization(network, simulation_data, i)
                time.sleep(1)

        if visualize:
            plt.ioff()  # Turn off interactive mode
            plt.show()   # Keep the final plot open

        return simulation_data
        
    def evaluate_strategy(run_data):
        pass

    def setup_real_time_visualization(self, network):
        plt.ion()
        self.fig, (self.ax_network, self.ax_metrics) = plt.subplots(1, 2, figsize=(16, 6))
        self.pos = nx.spring_layout(network.graph, seed=42)
        
        # Initial network plot
        node_colors = self._get_node_colors(network)
        self.network_nodes = nx.draw_networkx_nodes(
            network.graph, self.pos, ax=self.ax_network,
            node_color=node_colors, node_size=200,
            cmap='coolwarm', vmin=0, vmax=1
        )
        nx.draw_networkx_edges(network.graph, self.pos, ax=self.ax_network, alpha=0.3)
        
        # Add colorbar
        plt.colorbar(self.network_nodes, ax=self.ax_network, label='Proportion of Misinformation')
        
        # Initialize metrics plot
        self.metrics_line_u, = self.ax_metrics.plot([], [], 'b-', label='Ūₙ', linewidth=2)
        self.metrics_line_s, = self.ax_metrics.plot([], [], 'g-', label='S̄ₙ', linewidth=2)
        
        # Set up axes
        self.ax_network.set_title('Network State - Step 0')
        self.ax_network.axis('off')
        
        self.ax_metrics.set_xlabel('Time Steps')
        self.ax_metrics.set_ylabel('Proportion')
        self.ax_metrics.set_title('Metrics Over Time')
        self.ax_metrics.legend()
        self.ax_metrics.set_ylim(0, 1)
        self.ax_metrics.grid(True, alpha=0.3)
        
        self.fig.tight_layout()
        plt.draw()
    
    def update_real_time_visualization(self, network, simulation_data, current_step):
        """Update the real-time visualization"""
        # Update network plot
        node_colors = self._get_node_colors(network)
        self.ax_network.clear()
        
        self.network_nodes = nx.draw_networkx_nodes(
            network.graph, self.pos, ax=self.ax_network,
            node_color=node_colors, node_size=200,
            cmap='coolwarm', vmin=0, vmax=1
        )
        nx.draw_networkx_edges(network.graph, self.pos, ax=self.ax_network, alpha=0.3)
        
        self.ax_network.set_title(f'Network State - Step {current_step + 1}')
        self.ax_network.axis('off')
        
        # Update metrics plot
        steps = range(current_step + 1)
        U_data = simulation_data[:current_step + 1, 0]
        S_data = simulation_data[:current_step + 1, 1]
        
        self.metrics_line_u.set_data(steps, U_data)
        self.metrics_line_s.set_data(steps, S_data)
        
        self.ax_metrics.set_xlim(0, len(simulation_data))
        self.ax_metrics.set_ylim(0, 1)
        
        # Refresh display
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def _get_node_colors(self, network):
        return [network.nodes[node_id].get_proportion() 
                for node_id in network.graph.nodes()]

# === Initialize and Run Simulation
simulationRunner = SimulationRunner()
simulation_data = simulationRunner.run_simulation("erdos renyi", 100, 1)
print(simulation_data)