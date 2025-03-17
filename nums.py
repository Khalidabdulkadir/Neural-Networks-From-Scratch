import numpy as np

def Sigmoid(z):
    return 1 / 1 + (np.exp(z))

# input data 
X = np.array([2, 3])
weights = np.array([0.5, -1.2])
bais = 0.8

# Compute weights sum
Fx = np.dot(weights, X) + bais
print("FX: ", Fx)

output = Sigmoid(Fx)
print(output)