import numpy as np

class Perceptron:
    def __init__(self, threshold=0.5):
        self.threshold = threshold  

    def perceptron(self, inputs, weights):
        """
        A perceptron function that takes inputs and weights and uses a sigmoid activation.
        """
        if len(inputs) != len(weights):
            raise ValueError("Inputs and weights must be of the same length")

        weighted_sum = sum(i * w for i, w in zip(inputs, weights))
        print(f"Weighted Sum: {weighted_sum}")

        activation = self.sigmoid(weighted_sum)
        print(f"Activation Output: {activation}")

        if activation >= self.threshold:
            print("Yes, you can go to the party")
        else:
            print("No, you cannot go to the party")

    def sigmoid(self, z):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-z))

perceptron = Perceptron()
inputs = [1, 0, 1, 0, 1]
weights = [0.7, 0.6, 0.5, 0.3, 0.9]
perceptron.perceptron(inputs, weights)
