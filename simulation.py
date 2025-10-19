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


# === Declarations ===
class Node:
    def __init__(self, node_id, initial_red=1, initial_black=1, delta_r=1, delta_b=1, memory_size=10):
        self.node_id = node_id
        self.urn_red = initial_red
        self.urn_black = initial_black
        self.delta_r = delta_r
        self.delta_b = delta_b
        self.memory_size = memory_size
        self.history = []  # Will store last M draws

    def get_proportion(self): # Returns the Proportion of Red Balls in the Urn
        return self.urn_red / (self.urn_red + self.urn_black)
    
    def update(self, draw_result): # Updates the Contents of the Urn
        # Add new balls
        if draw_result == 1:
            self.urn_red += 1
        else:
            self.urn_black += 1

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
    def run_simulation(self, network_type, num_steps):
        if network_type == "erdos renyi":
            network = Network()
            network.initialize_erdos_renyi(100, 0.2)
        simulation_data = np.zeros((num_steps, 2))
        for i in range(num_steps):
            network.simulate_step()            
            U_bar, S_bar = network.get_network_metrics()
            simulation_data[i, 0] = U_bar
            simulation_data[i, 1] = S_bar
        return simulation_data
        
    def evaluate_strategy(run_data):
        pass


# === Initialize and Run Simulation
simulationRunner = SimulationRunner()
simulation_data = simulationRunner.run_simulation("erdos renyi", 100)
print(simulation_data)