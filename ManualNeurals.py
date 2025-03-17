import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Input
x = np.array([[200, 17.0]])  # Shape (1, 2)

# First neuron
w1_1 = np.array([1, 2])      # Shape (2,)
b1_1 = -1                    # Bias as a scalar
z1_1 = np.dot(x, w1_1) + b1_1  # Dot product + bias, shape (1,)
a1_1 = sigmoid(z1_1)         # Shape (1,)

# Second neuron
w1_2 = np.array([3, 4])      # Shape (2,)
b1_2 = 1                     # Bias as a scalar
z1_2 = np.dot(x, w1_2) + b1_2
a1_2 = sigmoid(z1_2)

# Third neuron
w1_3 = np.array([5, -6])     # Shape (2,)
b1_3 = 2
z1_3 = np.dot(x, w1_3) + b1_3
a1_3 = sigmoid(z1_3)

# Combine activations into a single array
a1 = np.array([a1_1.flatten(), a1_2.flatten(), a1_3.flatten()])

print("a1:", a1)
