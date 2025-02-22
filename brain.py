# brain.py
import numpy as np


class Brain:
    """
    Represents the neural network of a creature.
    This simple implementation uses a linear transformation.
    """

    def __init__(self, input_size: int, output_size: int):
        self.input_size = input_size
        self.output_size = output_size
        # Initialize random weights.
        self.weights = np.random.randn(input_size, output_size)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Computes the output (decision) from the input vector.
        """
        return np.dot(inputs, self.weights)
