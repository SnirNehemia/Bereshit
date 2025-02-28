# brain.py
import numpy as np
from typing import Callable, Optional


# TODO: pay attention to activation function and run as main to check the hidden layers
# Activation functions
def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


ACTIVATION_FUNCTIONS = {'relu': relu, 'sigmoid': sigmoid, 'tanh': tanh}


class Brain:
    def __init__(self, layers_size: list, activation: str = 'tanh', no_lineage=1, memory_nodes_count=0):
        """
        :param layers_size:
        :param activation:
        :param no_lineage:  indicate if it is a descendent to determine if it needs to reboot the brain
        :param memory_nodes:  values of memory nodes, should initiate with zeros
        """
        self.memory_nodes = np.zeros(memory_nodes_count)  # memory_nodes
        self.input_size = layers_size[0] + len(self.memory_nodes)
        self.output_size = layers_size[-1] + len(self.memory_nodes)
        self.activations = []  # Activation functions for each layer

        # self.memory_values = np.zeros(layers_size[-1])
        # self.activation = ACTIVATION_FUNCTIONS.get(activation)
        self.size = 0  # Effective network size
        self.random_magnitude = 0.1
        self.layers = []
        # Initialize a simple two-layer network
        if no_lineage:
            weight = np.random.randn(self.input_size, self.output_size) * self.random_magnitude
            self.layers.insert(0, weight)
            self.activations.insert(0, ACTIVATION_FUNCTIONS.get(activation))
            if len(layers_size) > 2:
                for i in range(1, len(layers_size) - 1):
                    self.add_layer(i, layers_size[i], activation)
        else:
            raise ValueError('No lineage')

    def add_layer(self, index: int, activation: str = 'relu'):
        if index < 0 or index >= len(self.layers):
            # index = len(self.layers)
            raise ValueError('Invalid layer index')

        input_size = self.layers[index - 1].shape[1] if index > 0 else self.input_size
        weight = np.eye(input_size)

        self.layers.insert(index, weight)
        self.activations.insert(index, ACTIVATION_FUNCTIONS.get(activation, sigmoid))
        # if index < 0 or index > len(self.layers):
        #     # index = len(self.layers)
        #     raise ValueError('Invalid layer index')
        #
        # input_size = self.layers[index - 1].shape[1] if index > 0 else self.input_size
        # weight = np.random.randn(input_size, size) * self.random_magnitude
        #
        # self.layers.insert(index, weight)
        # self.activations.insert(index, ACTIVATION_FUNCTIONS.get(activation, relu))
        #
        # # Adjust output connections of the new layer to match the next layer
        # if index < len(self.layers) - 1:
        #     next_layer = self.layers[index + 1]
        #     new_input_size = size
        #     new_next_layer = np.random.randn(new_input_size, next_layer.shape[1]) * self.random_magnitude
        #     self.layers[index + 1] = new_next_layer

        self.update_size()

    def add_neuron(self, index: int):  # 0 <= index< len(layers)
        if 0 <= index < len(self.layers) - 1:
            self.layers[index] = np.hstack((self.layers[index], np.zeros((self.layers[index].shape[0], 1))))
            self.layers[index + 1] = np.vstack(
                (self.layers[index + 1], np.random.randn(1, self.layers[index + 1].shape[1]) * self.random_magnitude))
            self.update_size()
        else:
            raise ValueError('Invalid layer index')

    def remove_neuron(self, index: int):  # 0 <= index< len(layers)
        if 0 <= index < len(self.layers) - 1:
            self.layers[index] = np.delete(self.layers[index], -1, axis=1)
            self.layers[index + 1] = np.delete(self.layers[index + 1], -1, axis=0)
            self.update_size()
        else:
            raise ValueError('Invalid layer index')

    # def remove_layer(self, index: int): # not in use
    #     if 0 <= index < len(self.layers):
    #         self.layers.pop(index)
    #         self.activations.pop(index)
    #         self.update_size()

    def change_layer(self, layer_index: int):
        if 0 <= layer_index < len(self.layers):
            self.layers[layer_index] += np.random.randn(
                # self.layers[layer_index].shape) * self.random_magnitude
                self.layers[layer_index].shape[0], self.layers[layer_index].shape[1]) * self.random_magnitude
        else:
            raise ValueError('Invalid layer index')

    def add_memory_node(self):
        self.memory_nodes = np.hstack((self.memory_nodes, np.zeros(1)))
        self.layers[0] = np.vstack(
            (self.layers[0], np.random.randn(1, self.layers[0].shape[1]) * self.random_magnitude))
        self.layers[-1] = np.hstack((self.layers[-1], np.zeros((self.layers[-1].shape[0], 1))))
        self.update_size()

    def remove_memory_node(self):
        if len(self.memory_nodes) > 0:
            self.memory_nodes = np.delete(self.memory_nodes, -1)
            self.layers[-1] = np.delete(self.layers[-1], -1, axis=1)
            self.layers[0] = np.delete(self.layers[0], -1, axis=0)
            self.update_size()

    def set_activation(self, layer_index: int, activation: str):
        if 0 <= layer_index < len(self.activations):
            self.activations[layer_index] = ACTIVATION_FUNCTIONS.get(activation, relu)

    def update_size(self):
        self.size = sum(layer.size for layer in self.layers)

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        x = np.concatenate([input_data, self.memory_nodes])
        self.neuron_values = [x]
        for weight, activation in zip(self.layers, self.activations):
            x = activation(x @ weight)
            self.neuron_values.append(x)
        if len(self.memory_nodes) > 0:
            self.memory_nodes = x[-len(self.memory_nodes):]
            return x[:-len(self.memory_nodes)]
        else:
            return x

    def mutate_brain(self, brain_mutation_rate: dict):  # not in use
        # mutation_roll = np.random.rand(len(brain_mutation_rate))
        # if mutation_roll[0] < brain_mutation_rate['layer_addition']:
        #     index = np.random.randint(0, len(self.layers))
        #     self.add_layer(index)
        # if mutation_roll[1] < brain_mutation_rate['modify_weights']:
        #     index = np.random.randint(0, len(self.layers))
        #     self.change_connectivity(index)
        # if mutation_roll[2] < brain_mutation_rate['modify_layer']:
        #     index = np.random.randint(0, len(self.layers))
        #     if np.random.rand() < 0.5:
        #         self.add_neuron(index)
        #     else:
        #         self.remove_neuron(index)
        return self


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def visualize_brain(brain, ax):
    ax.clear()

    # Set up the color map: blue for negative, white for zero, red for positive
    cmap = plt.cm.bwr
    norm = mcolors.Normalize(vmin=-1, vmax=1)

    layer_positions = []
    max_neurons = max(layer.shape[0] for layer in brain.neuron_values)

    # Draw neurons
    for i, neuron in enumerate(brain.neuron_values):
        layer_neurons = neuron.shape[0]
        y_positions = np.linspace(-max_neurons / 2, max_neurons / 2, layer_neurons)
        x_position = i * 2
        layer_positions.append((x_position, y_positions))

        # Draw neurons with color representing their value
        for y, value in zip(y_positions, brain.neuron_values[i]):
            ax.scatter(x_position, y, color=cmap(norm(value)), edgecolors='k', s=100, zorder=3)

    # Draw weights (connections between neurons)
    for i, weights in enumerate(brain.layers):
        x_start, y_start = layer_positions[i]
        x_end, y_end = layer_positions[i + 1]
        # weights = next_layer  # Assuming the weights are stored in the 'next' layer

        for j, start_y in enumerate(y_start):
            for k, end_y in enumerate(y_end):
                weight = weights.T[k, j] if weights.ndim > 1 else weights[j]
                ax.plot([x_start, x_end], [start_y, end_y], color=cmap(norm(weight)), zorder=1)

    ax.axis('equal')
    ax.axis('off')
    plt.draw()


if __name__ == '__main__':
    import matplotlib
    import platform
    if platform.system() == 'Darwin':
        matplotlib.use('MacOSX')
    else:
        matplotlib.use('TkAgg')
    np.random.seed(0)
    # Example usage
    brain = Brain([3, 5])
    brain.layers[0][0, 0] = 1
    brain.layers[0][0, 1] = -1
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
    brain.add_memory_node()  # still doesn't work
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
    brain.change_layer(1)
    brain.change_layer(2)
    # print('Brain output')
    # print(brain.forward(np.ones(3)))
    # brain.set_activation(1, 'tanh')
    output = brain.forward(np.random.rand(3))
    print('Output:', output)
    print('Effective size:', brain.size)
    fig, ax = plt.subplots()
    visualize_brain(brain, ax)
    plt.show()

