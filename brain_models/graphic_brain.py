import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch, Circle
import platform, matplotlib
import random


if platform.system() == 'Darwin':
    matplotlib.use('MacOSX')
else:
    matplotlib.use('TkAgg')

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def identity(x: np.ndarray) -> np.ndarray:
    return x

ACTIVATION_FUNCTIONS = {'relu': relu, 'sigmoid': sigmoid, 'tanh': tanh, 'I': identity}


class Brain:
    def __init__(self, layers_size: list, activation = 'tanh'):
        # Create a directed graph.
        self.input_counter = 0
        self.output_counter = 0
        self.hidden_counter = 0
        input_size = layers_size[0]
        output_size = layers_size[1]
        self.graph = nx.DiGraph()
        for i_in in range(input_size):
            self.add_node('input', 'I')
        for i_out in range(output_size):
            self.add_node( 'output', activation)
        for i_in in range(input_size):
            for i_out in range(output_size):
                self.add_connection(f'I{i_in}', f'O{i_out}', weight=np.random.randn())
        self.pos = 'none'
        self.random_magnitude = 0.2  # TODO: change to something from config

    def add_node(self, node_type='hidden', activation='tanh'):
        """Add a node with a given type (input, output, or hidden) and initial value."""
        self.pos = 'none'
        if node_type[0].capitalize() == 'H':
            ind = self.hidden_counter
        elif node_type[0].capitalize() == 'I':
            ind = self.input_counter
        elif node_type[0].capitalize() == 'O':
            ind = self.output_counter
        else:
            raise 'invalid node type'
        node_name = f'{node_type[0].capitalize()}{ind}'
        if node_name in self.graph:
            print(f"Node '{node_name}' already exists.")
        else:
            self.graph.add_node(node_name, type=node_type, activation=activation, value=0.0)
        if node_type[0].capitalize() == 'H':
            self.hidden_counter += 1
        elif node_type[0].capitalize() == 'I':
            self.input_counter += 1
        elif node_type[0].capitalize() == 'O':
            self.output_counter += 1

    def remove_node(self, node_id):
        """Remove a node and all its connections."""
        self.pos = 'none'
        if node_id in self.graph:
            self.graph.remove_node(node_id)
        else:
            print(f"Node '{node_id}' does not exist.")

    def add_connection(self, from_node, to_node, weight=0.0):
        """Add a directed connection with a weight from one node to another."""
        if from_node not in self.graph:
            print(f"From-node '{from_node}' does not exist.")
            return
        if to_node not in self.graph:
            print(f"To-node '{to_node}' does not exist.")
            return
        self.graph.add_edge(from_node, to_node, weight=weight)

    def remove_connection(self, from_node, to_node):
        """Remove the connection from one node to another."""
        if self.graph.has_edge(from_node, to_node):
            self.graph.remove_edge(from_node, to_node)
        else:
            print(f"Connection from '{from_node}' to '{to_node}' does not exist.")

    def adjust_weight(self, from_node, to_node):
        """Adjust the weight of an existing connection."""
        if self.graph.has_edge(from_node, to_node):
            self.graph[from_node][to_node]['weight'] += np.random.randn() * self.random_magnitude
        else:
            print(f"Connection from '{from_node}' to '{to_node}' does not exist.")

    def set_activation(self, node_id, activation):
        if node_id in self.graph:
            self.graph.nodes[node_id]['activation'] = activation
        else:
            print(f"Node '{node_id}' does not exist.")

    def forget(self, forget_magnitude=0):
        # forget_magnitude is the amplitude of forget
        if forget_magnitude == 0:
            for node, data in self.graph.nodes(data=True):
                if 'value' in data:
                    data['value'] = 0
        else:
            for node, data in self.graph.nodes(data=True):
                if 'value' in data:
                    data['value'] /= forget_magnitude

    def mutate(self, brain_mutation_rate: dict):
        # mutation_rate = {'add_node': 0.7, 'remove_node': 0.1, 'modify_edges': 0.7,
        # 'add_edge': 0.2, 'remove_edge': 0.1, 'change_activation': 0.1}
        mutation_roll = np.random.rand(len(brain_mutation_rate))
        self.pos = 'none'

        if mutation_roll[0] < brain_mutation_rate['add_node']:
            self.add_node('hidden', 'tanh')

        if mutation_roll[1] < brain_mutation_rate['remove_node']:
            ind = np.random.choice(len(self.graph.nodes))
            node_name = list(self.graph.nodes)[ind]
            if node_name[0].capitalize() == 'H':
                self.remove_node(node_name)

        if mutation_roll[2] < brain_mutation_rate['modify_edges']:
            ind = np.random.choice(len(self.graph.edges))
            self.adjust_weight(list(self.graph.edges)[ind][0], list(self.graph.edges)[ind][1])

        if mutation_roll[3] < brain_mutation_rate['add_edge']:
            # Build the set of all possible directed edges (excluding self-loops if desired)
            all_possible = {(i, o) for i in self.graph.nodes for o in self.graph.nodes
                            if i != o or not i[0] == 'O' or not o[0] == 'I'} # ignore possible input\output non sensible connections
            # Get the set of existing edges
            existing = set(self.graph.edges())
            # Missing edges are those in 'all_possible' but not in 'existing'
            missing = list(all_possible - existing)
            if len(missing) > 0:
                self.add_connection(*random.choice(missing))
            else:
                print('all connections are there!')

        if mutation_roll[4] < brain_mutation_rate['remove_edge']:
            if len(list(self.graph.edges)) > 2:
                ind = np.random.choice(len(self.graph.edges))
                connection = list(self.graph.edges)[ind]
                print(connection)
                # if not (connection[0][0] == 'I' and connection[1][0] == 'O'):  # to make sure the input has direct path to output
                self.remove_connection(*connection)
            else:
                print('too few connections to disconnect')

        if mutation_roll[5] < brain_mutation_rate['change_activation']:
            ind = np.random.choice(len(self.graph.nodes))
            node_name = list(self.graph.nodes)[ind]
            if node_name[0].capitalize() == 'H':
                self.set_activation(node_name, np.random.choice(list(ACTIVATION_FUNCTIONS.keys())))

        self.forget()
        return self

    def forward(self, start='input', activation_func=np.tanh):
        """
        Perform one pass along each connection, updating nodes in a synchronous update.

        Parameters:
            start (str): If 'input', use the current values of input nodes and update all others
                         based on the sum of (predecessor_value * weight). If 'output', use the current
                         values of output nodes and update all others based on successors.
            activation_func (callable): Function applied to the weighted sum (default is np.tanh).

        Returns:
            dict: If start=='input', returns the updated output node values;
                  if start=='output', returns the updated input node values.
        """
        if start not in ('input', 'output'):
            print("Invalid start parameter. Use 'input' or 'output'.")
            return None

        new_values = {}
        # Keep input nodes fixed; update every other node based on their predecessors.
        for node in self.graph.nodes():
            if self.graph.nodes[node].get('type') == 'input':
                new_values[node] = self.graph.nodes[node]['value']
            else:
                if self.graph.nodes[node].get('type') == 'output':
                    total = 0.0
                else:
                    total = self.graph.nodes[node]['value']  # to remain memory. alternatively, set to zero or multiply by forget coefficient (<1)
                for pred in self.graph.predecessors(node):
                    total += self.graph.nodes[pred]['value'] * self.graph[pred][node]['weight']
                new_values[node] = activation_func(total)

        # Update all nodes with their new computed values.
        for node, val in new_values.items():
            self.graph.nodes[node]['value'] = val

        # Return the values of output nodes.
        return {n: self.graph.nodes[n]['value']
                for n, d in self.graph.nodes(data=True) if d.get('type') == 'output'}

    def plot(self, ax='none', debug=False, plot_activatino=True):
        """
        Plot the brain graph:
          - Nodes are drawn as circles with colors according to their current value.
          - Connections are drawn as arrows colored on a red-white-blue scale (red: negative, blue: positive).
        """
        if debug:
            import matplotlib
            matplotlib.use('TkAgg')
            plt.ion()
            fig, ax = plt.subplots(figsize=(8, 6))
        ax.clear()
        if self.pos == 'none':
            self.pos = nx.spring_layout(self.graph, seed=1)  # Consistent layout
        # pos = nx.nx_agraph.graphviz_layout(self.graph, prog="dot")
        # Setup colormap for nodes.
        node_abs_max = max([abs(self.graph.nodes[n]['value']) for n in self.graph.nodes()])
        if node_abs_max:
            norm = Normalize(vmin=-node_abs_max, vmax=node_abs_max)
        else:
            norm = Normalize(vmin=-1, vmax=1)
        node_cmap = plt.get_cmap('bwr_r')  # Reversed so that negative=red, positive=blue

        # Draw nodes as circles.
        color_dict = {'input': 'black', 'hidden': 'none', 'output': 'darkgreen'}
        for node, (x, y) in self.pos.items():
            val = self.graph.nodes[node]['value']
            color = node_cmap(norm(val))
            # circle = Circle((x, y), radius=0.05, facecolor=color, alpha=0.5,
            #                 edgecolor=color_dict[self.graph.nodes[node].get('type')], linewidth=2.5, zorder=3)
            # ax.add_patch(circle)
            ax.scatter(x,y,s=300, facecolor=color, alpha=0.5, edgecolor=color_dict[self.graph.nodes[node].get('type')],
                       linewidth=2.5, zorder=3)
            if not self.graph.nodes[node].get('type') == 'hidden':
                ax.scatter(x, y, s=600, facecolor='none', edgecolor=color_dict[self.graph.nodes[node].get('type')],
                           linewidth=2.5, zorder=3)
            ax.text(x, y, str(np.round(self.graph.nodes[node]['value'],1)),
                    fontsize=8, ha='center', va='center', zorder=4)
            # ax.text(x, y, self.graph.nodes[node]['activation'],
            #         fontsize=8, ha='center', va='top', zorder=4)
            if plot_activatino:
                ax.annotate(self.graph.nodes[node]['activation'], xy=(x, y),
                         xytext=(0, -30), textcoords="offset points", ha='center')

        # Setup colormap for edges.
        weights_abs_max = max([abs(self.graph[u][v]['weight']) for u, v in self.graph.edges()])
        if weights_abs_max:
            w_norm = Normalize(vmin=-weights_abs_max, vmax=weights_abs_max)
        else:
            w_norm = Normalize(vmin=-1, vmax=1)
        edge_cmap = plt.get_cmap('bwr_r')

        # Draw edges with arrowheads.
        for u, v in self.graph.edges():
            start_point = self.pos[u]
            end_point = self.pos[v]
            weight = self.graph[u][v]['weight']
            color = edge_cmap(w_norm(weight))
            if (v, u) in self.graph.edges():
                rad = 0.4
                # if str(u) < str(v):
                #     rad = 0.2
                # else:
                #     rad = -0.2
            else:
                rad = 0.2
            arrow = FancyArrowPatch(start_point, end_point,
                                    arrowstyle='-|>', mutation_scale=15,
                                    color=color, lw=2, zorder=2,
                                    connectionstyle=f'arc3,rad={rad}',  # This sets the curvature
                                    shrinkA=10,  # Pull back from start point by x points TODO: Modify with network size?
                                    shrinkB=20  # Pull back from end point by x points
                                    )
            ax.add_patch(arrow)

        ax.set_aspect('equal')
        plt.axis('off')
        plt.title("Brain Graph")
        ax.relim()
        ax.margins(0.15)
        ax.autoscale_view(tight=True)
        plt.show()

    def __str__(self):
        nodes_str = ", ".join([f"{n} ({d['type']}, value={d.get('value', 0.0):.2f})"
                               for n, d in self.graph.nodes(data=True)])
        edges_str = ", ".join([f"{u}->{v}: {d['weight']}"
                               for u, v, d in self.graph.edges(data=True)])
        return f"Nodes: {nodes_str}\nConnections: {edges_str}"


# Example usage:
if __name__ == '__main__':
    brain = Brain([2,5])

    # Add nodes.
    # brain.add_node("I1", "input")
    # brain.add_node("I2", "input")
    brain.add_node("hidden")
    brain.add_node("hidden")
    brain.add_node("hidden")
    # brain.add_node("O1", "output")
    #
    # # Add connections.
    brain.add_connection("I1", "H1", weight=0.5)
    brain.add_connection("I2", "H1", weight=-0.8)
    brain.add_connection("H1", "O1", weight=-1.2)
    brain.add_connection("H1", "H2", weight=np.random.randn())
    brain.add_connection("I1", "O1", weight=np.random.randn())
    brain.add_connection("H1", "H0", weight=np.random.randn())
    # # Introduce a cycle (for demonstration).
    # brain.add_connection("H2", "H1", weight=0.3)

    # Set initial values for a forward pass starting from inputs.
    brain.graph.nodes["I0"]['value'] = 1.0
    brain.graph.nodes["I1"]['value'] = 0.5
    print("Forward pass (starting from inputs):")
    outputs = brain.forward(start='input')
    print("Output node values:", outputs)
    # Plot the brain.
    brain.plot(debug=True)
    # # Now set an initial value for outputs and propagate backward.
    # brain.graph.nodes["O1"]['value'] = 1.0
    # print("\nForward pass (starting from outputs):")
    # inputs = brain.forward(start='output')
    # print("Input node values:", inputs)

    # second forward pass
    brain.graph.nodes["I0"]['value'] = 1.0
    brain.graph.nodes["I1"]['value'] = 0.5
    # brain.remove_node('H1')
    print("Forward pass (starting from inputs):")
    outputs = brain.forward(start='input')
    print("Output node values:", outputs)
    brain.plot(debug=True)
