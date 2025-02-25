# brain.py
import numpy as np
from typing import Callable, Optional

# Activation functions
def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

ACTIVATION_FUNCTIONS = {'relu': relu, 'sigmoid': sigmoid, 'tanh': tanh}

class Brain:
    def __init__(self, layers_size: list, activation: str = 'relu'):
        self.input_size = layers_size[0]
        self.output_size = layers_size[-1]
        self.masks = []  # Connectivity masks
        self.activations = []  # Activation functions for each layer
        self.memory_nodes = []  # Indices of memory nodes
        self.memory_values = np.zeros(layers_size[-1])
        self.activation = ACTIVATION_FUNCTIONS.get(activation, relu)
        self.size = 0  # Effective network size
        self.layers = []
        if len(layers_size) > 2:
            for i in range(1, len(layers_size) - 1):
                self.add_layer(i, layers_size[i], activation)
        else:
            # Initialize a simple two-layer network
            self.add_layer(0, layers_size[-1], activation)



    def add_layer(self, index: int, size: int, activation: str = 'relu', mask: Optional[np.ndarray] = None):
        if index < 0 or index > len(self.layers):
            index = len(self.layers)

        input_size = self.layers[index - 1].shape[1] if index > 0 else self.input_size
        weight = np.random.randn(input_size, size) * 0.1
        if mask is None:
            mask = np.ones((input_size, size))

        self.layers.insert(index, weight)
        self.masks.insert(index, mask)
        self.activations.insert(index, ACTIVATION_FUNCTIONS.get(activation, relu))

        # Adjust output connections of the new layer to match the next layer
        if index < len(self.layers) - 1:
            next_layer = self.layers[index + 1]
            next_mask = self.masks[index + 1]
            new_input_size = size
            new_next_layer = np.random.randn(new_input_size, next_layer.shape[1]) * 0.1
            new_next_mask = np.ones((new_input_size, next_layer.shape[1]))
            self.layers[index + 1] = new_next_layer
            self.masks[index + 1] = new_next_mask

        self.update_size()

    def remove_layer(self, index: int):
        if 0 <= index < len(self.layers):
            self.layers.pop(index)
            self.masks.pop(index)
            self.activations.pop(index)
            self.update_size()

    def change_connectivity(self, layer_index: int, new_mask: np.ndarray):
        if 0 <= layer_index < len(self.masks):
            self.masks[layer_index] = new_mask

    def remove_connectivity(self, layer_index: int, indices: np.ndarray):
        if 0 <= layer_index < len(self.masks):
            self.masks[layer_index][indices] = 0

    def add_memory_node(self, node_index: int):
        if 0 <= node_index < self.output_size:
            self.memory_nodes.append(node_index)
            self.memory_nodes = list(set(self.memory_nodes))

    def remove_memory_node(self, node_index: int):
        if node_index in self.memory_nodes:
            self.memory_nodes.remove(node_index)

    def set_activation(self, layer_index: int, activation: str):
        if 0 <= layer_index < len(self.activations):
            self.activations[layer_index] = ACTIVATION_FUNCTIONS.get(activation, relu)

    def update_size(self):
        self.size = sum(np.sum(mask) for mask in self.masks)

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        x = np.concatenate([input_data, self.memory_values[self.memory_nodes]])
        for weight, mask, activation in zip(self.layers, self.masks, self.activations):
            x = activation(x @ (weight * mask))
        self.memory_values[self.memory_nodes] = x[self.memory_nodes]
        return x
if __name__ == '__main__':
    # Example usage
    brain = Brain(input_size=10, output_size=5)
    brain.add_layer(1, 3, 'tanh')
    brain.add_memory_node(2)
    brain.set_activation(1, 'tanh')
    output = brain.forward(np.random.rand(10))
    print('Output:', output)
    print('Effective size:', brain.size)

