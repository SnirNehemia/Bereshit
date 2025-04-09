import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch, Circle
import platform, matplotlib
import random, math
from config import Config as config

# TODO: make closed loop activation tanh! or regulate it

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
        self.self_connected = []
        self.size = len(self.graph.nodes)

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
            self.graph.add_node(node_name, type=node_type, value=0.0)
            self.set_activation(node_name, activation)
        if node_type[0].capitalize() == 'H':
            self.hidden_counter += 1
        elif node_type[0].capitalize() == 'I':
            self.input_counter += 1
        elif node_type[0].capitalize() == 'O':
            self.output_counter += 1
        return node_name

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

    def break_connection(self, from_node, to_node):
        """
        Break the connection from one node to another by inserting a new hidden node in between.

        The weight of the old connection is split between the new connections.
        """
        if self.graph.has_edge(from_node, to_node):
            old_weight = self.graph[from_node][to_node]['weight']
            new_weight = old_weight / 2 + np.random.randn()*old_weight/5
            new_node = self.add_node('hidden', 'tanh')
            self.remove_connection(from_node, to_node)
            self.add_connection(from_node, new_node, weight=new_weight)
            self.add_connection(new_node, to_node, weight=old_weight-new_weight)

    def adjust_weight(self, from_node, to_node):
        """Adjust the weight of an existing connection."""
        if self.graph.has_edge(from_node, to_node):
            self.graph[from_node][to_node]['weight'] += np.random.randn() * self.random_magnitude
        else:
            print(f"Connection from '{from_node}' to '{to_node}' does not exist.")

    def set_activation(self, node_id, activation):
        if node_id in self.graph:
            self.graph.nodes[node_id]['activation_str'] = activation
            self.graph.nodes[node_id]['activation'] = ACTIVATION_FUNCTIONS.get(activation)
        else:
            print(f"Node '{node_id}' does not exist.")

    def forget(self, forget_magnitude=0):
        """
        Forget the values of all nodes in the network.

        If `forget_magnitude` is 0, all node values are set to 0.
        Otherwise, each node value is divided by `forget_magnitude`.

        Parameters
        ----------
        forget_magnitude : float, optional
            The amplitude of forgetting. If 0, all values are set to 0.
            Otherwise, each value is divided by this number.
        """
        if forget_magnitude == 0:
            for node, data in self.graph.nodes(data=True):
                if 'value' in data:
                    data['value'] = 0
        else:
            for node, data in self.graph.nodes(data=True):
                if 'value' in data:
                    data['value'] /= forget_magnitude

    def mutate(self, brain_mutation_rate: dict):
        # mutation_rate = {'add_node': 0.7, 'remove_node': 0.1, 'modify_edges': 0.7, 'modify_edges_percentage': 0.5,
        # 'add_edge': 0.2, 'remove_edge': 0.1, 'change_activation': 0.1, 'forget_magnitude': 10}
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
            indices = np.random.choice(len(self.graph.edges),
                                    math.ceil(brain_mutation_rate['modify_edges_percentage']*len(self.graph.edges)))
            if len(indices) > 0:
                for ind in indices:
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
                count = 0
                found_new_edge = False
                while not found_new_edge and count < 10:
                    count += 1
                    new_edge = random.choice(missing)
                    if not (new_edge[0][0] == 'O' or new_edge[1][0] == 'I'):
                        # print(new_edge)
                        self.add_connection(*new_edge, weight=np.random.randn())
                        found_new_edge = True
                if not found_new_edge:
                    pass
                    # print('did not found new edge')
            else:
                print('all connections are there!')

        if mutation_roll[4] < brain_mutation_rate['remove_edge']:
            if len(list(self.graph.edges)) > 2:
                ind = np.random.choice(len(self.graph.edges))
                connection = list(self.graph.edges)[ind]
                # print(connection)
                # if not (connection[0][0] == 'I' and connection[1][0] == 'O'):  # to make sure the input has direct path to output
                self.remove_connection(*connection)
            else:
                print('too few connections to disconnect')

        if mutation_roll[5] < brain_mutation_rate['change_activation']:
            ind = np.random.choice(len(self.graph.nodes))
            node_name = list(self.graph.nodes)[ind]
            if node_name[0].capitalize() == 'H':
                activation_set = np.random.choice(list(ACTIVATION_FUNCTIONS.keys()))
                while activation_set == 'I':
                    activation_set = np.random.choice(list(ACTIVATION_FUNCTIONS.keys()))
                self.set_activation(node_name, np.random.choice(list(ACTIVATION_FUNCTIONS.keys())))

        if mutation_roll[6] < brain_mutation_rate['add_loop']:
            node = np.random.choice(self.graph.nodes)
            if not node in self.self_connected and self.graph.nodes[node].get('type') == 'hidden':
                self.self_connected.append(node)
                self.add_connection(node, node, weight=np.random.randn())

        if mutation_roll[7] < brain_mutation_rate['break_edge']:
            if len(list(self.graph.edges)) > 2:
                ind = np.random.choice(len(self.graph.edges))
                connection = list(self.graph.edges)[ind]
                self.break_connection(*connection)

        self.forget(brain_mutation_rate[
                        'forget_magnitude'])  # reduce the current value of the nodes as a forget mechanism between generations
        self.size = len(self.graph.nodes)
        return self

    def normalize_input(self, input):
        input /= config.NORM_INPUT  # normalize the input with prior knowledge
        input = np.tanh(input)  # normalize the input further
        return input

    def forward(self, input, normalize=True):
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
        if normalize: input = self.normalize_input(input)
        for i in range(len(input)):
            self.graph.nodes[f'I{i}']['value'] = input[i]
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
                new_values[node] = total

        # Update all nodes with their new computed values.
        for node, val in new_values.items():
            self.graph.nodes[node]['value'] = self.graph.nodes[node]['activation'](val)

        # Return the values of output nodes.
        return [self.graph.nodes[n]['value']
                for n, d in self.graph.nodes(data=True) if d.get('type') == 'output']

    def get_pos(self, mode='default'):
        if mode == 'default' or mode == 'depth':
            # --- Calculate depth for each node based on the number of connections from an input node ---
            # For input nodes, depth is 0.
            input_nodes = [n for n, data in self.graph.nodes(data=True) if data.get("type") == "input"]
            for node in self.graph.nodes():
                if self.graph.nodes[node].get("type") == "input":
                    self.graph.nodes[node]["depth"] = 0
                else:
                    distances = []
                    # Compute shortest path length from each input node to the current node.
                    for inp in input_nodes:
                        try:
                            d = nx.shortest_path_length(self.graph, source=inp, target=node)
                            distances.append(d)
                        except nx.NetworkXNoPath:
                            continue
                    if distances:
                        self.graph.nodes[node]["depth"] = min(distances)
                    else:
                        # If no input node can reach this node, assign a default depth.
                        self.graph.nodes[node]["depth"] = None

            # --- Create a custom layout function ---
            def assign_positions(nodes, fixed_y):
                """Assigns x positions evenly across [0,1] for the given nodes at a fixed y-coordinate."""
                pos_layer = {}
                n = len(nodes)
                for i, node in enumerate(nodes):
                    x = (i + 1) / (n + 1)
                    pos_layer[node] = np.array([x, fixed_y])
                return pos_layer

            # Separate nodes by type.
            input_nodes = [n for n, data in self.graph.nodes(data=True) if data.get("type") == "input"]
            output_nodes = [n for n, data in self.graph.nodes(data=True) if data.get("type") == "output"]
            hidden_nodes = [n for n, data in self.graph.nodes(data=True) if data.get("type") == "hidden"]

            # Fix y coordinates for input (bottom) and output (top) nodes.
            pos_input = assign_positions(input_nodes, 0)
            pos_output = assign_positions(output_nodes, 1)

            # For hidden nodes, use the computed 'depth' to determine y, normalizing to a range (e.g., 0.3 to 0.7).
            pos_hidden = {}
            if hidden_nodes:
                # Extract depths (ignoring any that might be None).
                depths = [self.graph.nodes[n]["depth"] for n in hidden_nodes if self.graph.nodes[n]["depth"] is not None]
                if depths:
                    min_depth, max_depth = min(depths), max(depths)
                else:
                    min_depth, max_depth = 0, 1  # Fallback if all depths are None.
                y_min, y_max = 0.3, 0.7
                # Sort hidden nodes by depth for consistent x placement.
                hidden_nodes_sorted = sorted(hidden_nodes, key=lambda n: self.graph.nodes[n]["depth"] if self.graph.nodes[n][
                                                                                                    "depth"] is not None else 0.5)
                n_hidden = len(hidden_nodes_sorted)
                for i, node in enumerate(hidden_nodes_sorted):
                    depth = self.graph.nodes[node]["depth"] if self.graph.nodes[node]["depth"] is not None else (
                                                                                                          min_depth + max_depth) / 2
                    # Normalize depth within the chosen y range.
                    if max_depth > min_depth:
                        y = y_min + (depth - min_depth) / (max_depth - min_depth) * (y_max - y_min)
                    else:
                        y = (y_min + y_max) / 2
                    x = (i + 1) / (n_hidden + 1)
                    pos_hidden[node] = np.array([x, y])

            # Combine all positions.
            pos = {}
            pos.update(pos_input)
            pos.update(pos_hidden)
            pos.update(pos_output)
            # add weak spring layout
            temp_pos_graph = self.graph.copy()
            for u, v in temp_pos_graph.edges():
                temp_pos_graph[u][v]['weight'] = abs(temp_pos_graph[u][v]['weight'])
            pos = nx.spring_layout(temp_pos_graph, pos=pos, seed=42, k=0.5, iterations=1)
            self.pos = pos
        elif mode == 'spring':
            # Identify nodes by type.
            input_nodes = [n for n, d in self.graph.nodes(data=True) if d.get("type") == "input"]
            output_nodes = [n for n, d in self.graph.nodes(data=True) if d.get("type") == "output"]
            hidden_nodes = [n for n, d in self.graph.nodes(data=True) if d.get("type") == "hidden"]

            # Set initial positions.
            pos_initial = {}
            fixed_nodes = []

            # Place input nodes at y=0. Spread them along x.
            n_input = len(input_nodes)
            for i, node in enumerate(input_nodes):
                pos_initial[node] = (i / (n_input - 1) if n_input > 1 else 0.5, 0)
                fixed_nodes.append(node)

            # Place output nodes at y=1. Spread them along x.
            n_output = len(output_nodes)
            for i, node in enumerate(output_nodes):
                pos_initial[node] = (i / (n_output - 1) if n_output > 1 else 0.5, 1)
                fixed_nodes.append(node)

            # For hidden nodes, assign arbitrary initial positions (y around 0.5).
            for node in hidden_nodes:
                pos_initial[node] = (random.random(), 0.5)

            temp_pos_graph = self.graph.copy()
            for u, v in temp_pos_graph.edges():
                temp_pos_graph[u][v]['weight'] = abs(temp_pos_graph[u][v]['weight'])

            # Use spring_layout with fixed positions for inputs and outputs.
            self.pos = nx.spring_layout(temp_pos_graph, pos=pos_initial, seed=42, k=0.5)  # , fixed=fixed_nodes

    def plot(self, ax='none', debug=False, plot_activation=False, plot_values=False):
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
        ax.set_aspect(2)
        if self.pos == 'none':
            self.get_pos()  # Consistent layout
        # pos = nx.nx_agraph.graphviz_layout(self.graph, prog="dot")
        # Setup colormap for nodes.
        node_abs_max = max([abs(self.graph.nodes[n]['value']) for n in self.graph.nodes()]) # normalize node display
        if node_abs_max:
            norm = Normalize(vmin=-node_abs_max, vmax=node_abs_max)
        else:
            norm = Normalize(vmin=-1, vmax=1)
        node_cmap = plt.get_cmap('bwr_r')  # Reversed so that negative=red, positive=blue

        # Draw nodes as circles.
        color_dict = {'input': 'black', 'hidden': 'slategrey', 'output': 'gold'}
        for node, (x, y) in self.pos.items():
            val = self.graph.nodes[node]['value']
            color = node_cmap(norm(val))
            # circle = Circle((x, y), radius=0.05, facecolor=color, alpha=0.5,
            #                 edgecolor=color_dict[self.graph.nodes[node].get('type')], linewidth=2.5, zorder=3)
            # ax.add_patch(circle)
            ax.scatter(x,y,s=200, facecolor=color, alpha=1, edgecolor=color_dict[self.graph.nodes[node].get('type')],
                       linewidth=1, zorder=3)
            if not self.graph.nodes[node].get('type') == 'hidden':
                ax.scatter(x, y, s=400, facecolor='none', edgecolor=color_dict[self.graph.nodes[node].get('type')],
                           linewidth=1.5, zorder=4)
            if plot_values:
                ax.text(x, y, str(np.round(self.graph.nodes[node]['value'],1)),
                        fontsize=8, ha='center', va='center', zorder=4)
                # ax.text(x, y, self.graph.nodes[node]['activation'],
                #         fontsize=8, ha='center', va='top', zorder=4)
            if plot_activation:
                ax.annotate(self.graph.nodes[node]['activation_str'], xy=(x, y),
                         xytext=(0, -30), textcoords="offset points", ha='center')

        # Setup colormap for edges.
        weights_abs_max = 2 # max([abs(self.graph[u][v]['weight']) for u, v in self.graph.edges()])
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
            rad = np.tanh(weight) * 0.4 # weight
            # rad = 0.4 # + (np.random.random() - 0.5) * 0.1
            # if (v, u) in self.graph.edges():
            #     rad = np.random.random() - 0.5 # 0.4 + (np.random.random() - 0.5) * 0.4
            #     # if str(u) < str(v):
            #     #     rad = 0.2
            #     # else:
            #     #     rad = -0.2
            # else:
            #     rad = 0.2
            if np.linalg.norm(start_point-end_point) <= 1e-1:
                # For a self-loop, use a fixed curvature (rad) that draws a circular arc.
                # print(start_point)
                # offset = 10  # tweak as needed for your coordinate scale
                # start_point = (x - offset, y)
                # end_point = (x + offset, y)
                arrow = FancyArrowPatch(
                    start_point,
                    end_point,
                    arrowstyle='-|>',
                    mutation_scale=15,
                    color=color,
                    lw=2,
                    zorder=2,
                    connectionstyle="arc,angleA=45,angleB=90,armA=90,armB=90,rad=20",  # Fixed curvature for a self-loop
                    shrinkA=10,
                    shrinkB=10  # Use equal shrink to have the arrow touch the circle neatly
                )
            else:
                # For a regular edge, use a variable curvature (rad) that draws a smooth arc.
                arrow = FancyArrowPatch(
                    start_point,
                    end_point,
                    arrowstyle='-|>',
                    mutation_scale=15,
                    color=color,
                    lw=2,
                    zorder=2,
                    connectionstyle=f'arc3,rad={rad}',  # Use your variable rad for regular edges
                    shrinkA=10,
                    shrinkB=20,
                    alpha=np.clip(abs(weight),1e-2,1)
                )
            # arrow = FancyArrowPatch(start_point, end_point,
            #                         arrowstyle='-|>', mutation_scale=15,
            #                         color=color, lw=2, zorder=2,
            #                         connectionstyle=f'arc3,rad={rad}',  # This sets the curvature
            #                         shrinkA=10,  # Pull back from start point by x points TODO: Modify with network size?
            #                         shrinkB=20  # Pull back from end point by x points
            #                         )
            ax.add_patch(arrow)

        ax.set_aspect('equal')
        ax.set_title("Brain Graph")
        ax.relim()
        ax.autoscale_view(tight=True)
        ax.margins(0.2)
        ax.axis('off')
        if debug: plt.show()

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
    # brain.add_node("hidden")
    # brain.add_node("hidden")
    # brain.add_node("hidden")
    # brain.add_node("O1", "output")
    #
    # # Add connections.
    # brain.add_connection("I1", "H1", weight=0.5)
    # brain.add_connection("I2", "H1", weight=-0.8)
    # brain.add_connection("H1", "O1", weight=-1.2)
    # brain.add_connection("H1", "H2", weight=np.random.randn())
    # brain.add_connection("I1", "O1", weight=np.random.randn())
    # brain.add_connection("H1", "H0", weight=np.random.randn())
    # brain.add_connection("H1", "H2", weight=0.0001)
    brain.add_connection("H2", "O1", weight=-0.0001)
    # brain.add_connection("I1", "H3", weight=10)
    # brain.add_connection("H3", "H3", weight=10)
    # brain.add_connection("I2", "H3", weight=-10)
    # brain.add_connection("H1", "H1", weight=5)
    # brain.add_connection("O1", "O1", weight=-10)
    # brain.add_connection("H0", "H0", weight=20)
    # # Introduce a cycle (for demonstration).
    # brain.add_connection("H2", "H1", weight=0.3)
    for i in range(10):
        brain.mutate(config.MUTATION_GRAPH_BRAIN)
    brain.add_connection("H3", "H2", weight=-0.0001)
    # Set initial values for a forward pass starting from inputs.
    input = [1.0, 0.5]
    print("Forward pass (starting from inputs):")
    outputs = brain.forward(input, normalize=False)
    print("Output node values:", outputs)
    # Plot the brain.
    brain.plot(debug=True)
    # # Now set an initial value for outputs and propagate backward.
    # brain.graph.nodes["O1"]['value'] = 1.0
    # print("\nForward pass (starting from outputs):")
    # inputs = brain.forward(start='output')
    # print("Input node values:", inputs)

    # # second forward pass
    # brain.graph.nodes["I0"]['value'] = 1.0
    # brain.graph.nodes["I1"]['value'] = 0.5
    # brain.mutate(config.MUTATION_GRAPH_BRAIN)
    # brain.mutate(config.MUTATION_GRAPH_BRAIN)
    # brain.mutate(config.MUTATION_GRAPH_BRAIN)
    # brain.mutate(config.MUTATION_GRAPH_BRAIN)
    # brain.mutate(config.MUTATION_GRAPH_BRAIN)
    # brain.mutate(config.MUTATION_GRAPH_BRAIN)
    # # brain.remove_node('H1')
    # print("Forward pass (starting from inputs):")
    # outputs = brain.forward(input, normalize=False)
    # print("Output node values:", outputs)
    # brain.plot(debug=True)
