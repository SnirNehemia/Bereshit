# fully_connected_brain.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from input.codes import config

import torch
import torch.nn as nn
import torch.nn.functional as F

class Brain(nn.Module):
    # Methods we have to implement:
    #   mutate(self, input_data: np.ndarray)
    #   mutate(self, error: np.ndarray) while: brain_mutation_rate = {'layer_addition': 0.1, 'modify_weights': 0.1, 'modify_layer': 0.1, 'modify_activation': 0.2}
    #   plot(self, ax: matplotlib axis, debug: bool = False)

    def __init__(self, layers_size: list, activation: str = 'tanh', memory_nodes_count=0):
        super(Brain, self).__init__()

        # Initialize memory nodes as a buffer (not a parameter to be trained by SGD)
        self.register_buffer('memory_nodes', torch.zeros(memory_nodes_count))

        self.input_size = layers_size[0] + memory_nodes_count
        self.output_size = layers_size[-1] + memory_nodes_count
        self.random_magnitude = 0.2
        self.size = 0  # Effective network size
        self.layers = nn.ParameterList()
        self.activations_str = []

        self.neuron_values = []  # For plotting

        # Initialize first layer
        weight = torch.randn(self.output_size, self.input_size) * self.random_magnitude
        self.layers.append(nn.Parameter(weight))
        self.activations_str.append(activation)

        if len(layers_size) > 2:
            for i in range(1, len(layers_size) - 1):
                self.add_layer(i, activation)

    def get_activation_fn(self, name):
        mapping = {'relu': F.relu, 'sigmoid': torch.sigmoid, 'tanh': torch.tanh}
        return mapping.get(name, torch.tanh)

    @torch.no_grad()
    def add_layer(self, index: int, activation: str = 'tanh'):
        # In (out, in) format: in_features is dim 1 of the following layer
        # or dim 0 of the previous layer's output
        in_features = self.layers[index - 1].shape[0] if index > 0 else self.input_size
        weight = torch.eye(in_features)  # Identity matrix (out == in)

        new_params = []
        for i, p in enumerate(self.layers):
            if i == index:
                new_params.append(nn.Parameter(weight))
            new_params.append(p)
        if index == len(self.layers):
            new_params.append(nn.Parameter(weight))

        self.layers = nn.ParameterList(new_params)
        self.activations_str.insert(index, activation)
        self.update_size()

    @torch.no_grad()
    def add_neuron(self, index: int):
        if 0 <= index < len(self.layers) - 1:
            # Expand output of layer[index]: shape (out, in) -> (out+1, in)
            old_w1 = self.layers[index]
            new_row = torch.zeros(1, old_w1.shape[1])
            self.layers[index] = nn.Parameter(torch.cat([old_w1, new_row], dim=0))

            # Expand input of layer[index+1]: shape (out, in) -> (out, in+1)
            old_w2 = self.layers[index + 1]
            new_col = torch.randn(old_w2.shape[0], 1) * self.random_magnitude
            self.layers[index + 1] = nn.Parameter(torch.cat([old_w2, new_col], dim=1))

            self.update_size()
        else:
            raise ValueError('Invalid layer index')

    @torch.no_grad()
    def remove_neuron(self, index: int):
        if 0 <= index < len(self.layers) - 1:
            # Only remove if there's more than one neuron to keep architecture valid
            if self.layers[index].shape[0] > 1:
                # Slice out last row of current, last column of next
                self.layers[index] = nn.Parameter(self.layers[index][:-1, :])
                self.layers[index + 1] = nn.Parameter(self.layers[index + 1][:, :-1])

    def update_size(self):
        self.size = sum(layer.numel() for layer in self.layers)

    @torch.no_grad()
    def add_memory_node(self):
        # Update buffer
        self.memory_nodes = torch.cat([self.memory_nodes, torch.zeros(1)])

        # First Layer: add a column (increase input_size)
        old_first = self.layers[0]
        new_col = torch.randn(old_first.shape[0], 1) * self.random_magnitude
        self.layers[0] = nn.Parameter(torch.cat([old_first, new_col], dim=1))

        # Last Layer: add a row (increase output_size)
        old_last = self.layers[-1]
        new_row = torch.zeros(1, old_last.shape[1])
        self.layers[-1] = nn.Parameter(torch.cat([old_last, new_row], dim=0))

    @torch.no_grad()
    def remove_memory_node(self):
        if len(self.memory_nodes) > 0:
            self.memory_nodes = self.memory_nodes[:-1]
            # Shrink first layer input (dim 1) and last layer output (dim 0)
            self.layers[0] = nn.Parameter(self.layers[0][:, :-1])
            self.layers[-1] = nn.Parameter(self.layers[-1][:-1, :])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x /= config.NORM_INPUT.astype(np.float32)  # normalize the input with prior knowledge
        x = torch.tanh(x)  # Initial normalization
        x = torch.cat([x, self.memory_nodes])

        self.neuron_values = [x.detach().cpu().numpy()]

        for i, weight in enumerate(self.layers):
            act_fn = self.get_activation_fn(self.activations_str[i])
            x = act_fn(F.linear(x, weight))
            self.neuron_values.append(x.detach().cpu().numpy())

        if len(self.memory_nodes) > 0:
            mem_size = len(self.memory_nodes)
            self.memory_nodes = x[-mem_size:].detach()
            return x[:-mem_size]
        return x

    @torch.no_grad()
    def modify_layer(self, idx: int):
        if 0 <= idx < len(self.layers):
            # self.layers[index] = nn.Parameter(torch.randn_like(self.layers[index]) * self.random_magnitude)
            noise = torch.randn_like(self.layers[idx]) * self.random_magnitude
            self.layers[idx].add_(noise)
        else:
            raise ValueError('Invalid layer index')

    @torch.no_grad()
    def mutate(self, brain_mutation_rate: dict):
        mutation_roll = np.random.rand(4)

        # 1. Add layer
        if mutation_roll[0] < brain_mutation_rate.get('layer_addition', 0):
            idx = np.random.randint(0, len(self.layers))
            self.add_layer(idx)

        # 2. Modify weights (Add Gaussian noise)
        if mutation_roll[1] < brain_mutation_rate.get('modify_weights', 0):
            idx = np.random.randint(0, len(self.layers))
            # noise = torch.randn_like(self.layers[idx]) * self.random_magnitude
            # self.layers[idx].add_(noise)
            self.modify_layer(self, idx)

        # 3. Add/Remove neuron
        if mutation_roll[2] < brain_mutation_rate.get('modify_layer', 0) and len(self.layers) > 1:
            idx = np.random.randint(0, len(self.layers) - 1)
            self.add_neuron(idx)  # Simplified for brevity, add remove logic here if needed

        # 4. Change activation
        if mutation_roll[3] < brain_mutation_rate.get('modify_activation', 0):
            idx = np.random.randint(0, len(self.layers))
            self.activations_str[idx] = np.random.choice(['relu', 'sigmoid', 'tanh'])

        return self

    def plot(self, ax):
        ax.clear()
        cmap = plt.cm.bwr
        norm = mcolors.Normalize(vmin=-1, vmax=1)

        layer_positions = []
        max_neurons = max(len(n) for n in self.neuron_values)

        # Draw Neurons
        for i, neuron_vec in enumerate(self.neuron_values):
            n_count = len(neuron_vec)
            y_pos = np.linspace(-max_neurons / 2, max_neurons / 2, n_count)
            x_pos = i * 2
            layer_positions.append((x_pos, y_pos))

            for y, val in zip(y_pos, neuron_vec):
                ax.scatter(x_pos, y, color=cmap(norm(val)), edgecolors='k', s=100, zorder=3)

            label = 'input' if i == 0 else self.activations_str[i - 1]
            ax.text(x_pos, -max_neurons / 2 - 1, label, ha='center', fontsize=8)

        # Draw Weights (Connections)
        for i, param in enumerate(self.layers):
            # weight shape is (out, in)
            weight_mat = param.detach().cpu().numpy()
            x_start, y_start = layer_positions[i]
            x_end, y_end = layer_positions[i + 1]

            # In (out, in), weight_mat[k, j] connects neuron j of layer i to neuron k of layer i+1
            for j, start_y in enumerate(y_start):
                for k, end_y in enumerate(y_end):
                    w = weight_mat[k, j]
                    alpha = min(abs(w), 1.0)
                    if alpha > 0.1:  # Optimization: don't draw tiny weights
                        ax.plot([x_start, x_end], [start_y, end_y],
                                color=cmap(norm(w)), alpha=alpha, zorder=1)

        ax.axis('off')
        ax.set_title(f"PyTorch Brain - Params: {sum(p.numel() for p in self.layers)}")


if __name__ == '__main__':
    import matplotlib
    import platform
    if platform.system() == 'Darwin':
        matplotlib.use('MacOSX')
    else:
        matplotlib.use('TkAgg')

    # Load config
    config_yaml_relative_path = r"input\yamls\2025_06_20_config.yaml"
    config = config.load_config(yaml_relative_path=config_yaml_relative_path)

    # Example usage
    torch.manual_seed(0)
    eye_channel_count = 9
    brain = Brain([eye_channel_count, 5])

    brain.add_layer(0)  # 0<=X< len(layers)
    print('new_layer:', brain.layers[0].shape, brain.layers[1].shape)
    brain.remove_neuron(0)  # 0<=X< len(layers)
    print('after remove:', brain.layers[0].shape, brain.layers[1].shape)
    brain.add_neuron(0)
    print('after add:', brain.layers[0].shape, brain.layers[1].shape)
    brain.add_layer(0)
    print('after add layer:', brain.layers[0].shape, brain.layers[1].shape, brain.layers[2].shape)
    brain.remove_neuron(1)  # 0<=X< len(layers)
    print('after remove:', brain.layers[0].shape, brain.layers[1].shape, brain.layers[2].shape)
    brain.add_memory_node()
    print('after add memory:', brain.layers[0].shape, brain.layers[1].shape, brain.layers[2].shape)
    print(brain.memory_nodes.shape)
    brain.add_memory_node()  # still doesn't work
    print('after add memory:', brain.layers[0].shape, brain.layers[1].shape, brain.layers[2].shape)
    print(brain.memory_nodes.shape)
    brain.add_memory_node()  # still doesn't work
    print('after add memory:', brain.layers[0].shape, brain.layers[1].shape, brain.layers[2].shape)
    print(brain.memory_nodes.shape)
    brain.remove_memory_node()  # still doesn't work
    print('after remove memory:', brain.layers[0].shape, brain.layers[1].shape, brain.layers[2].shape)
    print(brain.memory_nodes.shape)
    brain.modify_layer(1)
    brain.modify_layer(2)
    brain.add_layer(1)
    brain.remove_neuron(2)
    brain.remove_neuron(2)
    brain.remove_neuron(1)
    brain.remove_neuron(1)
    brain.remove_neuron(1)
    brain.remove_neuron(1)
    brain.remove_neuron(1)
    brain.remove_neuron(0)
    brain.remove_neuron(0)
    # print('Brain output')
    # print(brain.forward(np.ones(3)))
    # brain.set_activation(1, 'tanh')
    output = brain.forward(torch.randn(eye_channel_count))
    print('Output:', output)
    print('Effective size:', brain.size)
    fig, ax = plt.subplots()
    brain.plot(ax)
    plt.show()

