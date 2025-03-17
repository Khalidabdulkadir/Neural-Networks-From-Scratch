import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        self.weights = np.zeros(input_size)
        self.bias = 0
        self.learning_rate = learning_rate

    def activation(self, z):
        return 1 if z >= 0 else 0

    def forward(self, inputs):
        z = np.dot(inputs, self.weights) + self.bias
        return self.activation(z)

    def train(self, training_data, labels, epochs=100):
        for epoch in range(epochs):
            for inputs, label in zip(training_data, labels):
                prediction = self.forward(inputs)
                error = label - prediction
                self.weights += self.learning_rate * error * inputs
                self.bias += self.learning_rate * error

training_data = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

labels = np.array([0, 0, 0, 1])

perceptron = Perceptron(input_size=2, learning_rate=0.1)

perceptron.train(training_data, labels, epochs=100)

print("Testing the Perceptron for AND Gate:")
for inputs in training_data:
    output = perceptron.forward(inputs)
    print(f"Input: {inputs} -> Output: {output}")