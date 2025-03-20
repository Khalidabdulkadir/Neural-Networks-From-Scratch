import numpy as np

# Mean Squared Error (MSE)
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Mean Absolute Error (MAE)
def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# Huber Loss (delta = 1)
def huber_loss(y_true, y_pred, delta=1.0):
    error = y_true - y_pred
    is_small_error = np.abs(error) <= delta
    squared_loss = 0.5 * error ** 2
    linear_loss = delta * (np.abs(error) - 0.5 * delta)
    return np.mean(np.where(is_small_error, squared_loss, linear_loss))

# Binary Cross-Entropy Loss
def binary_cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)  # Avoid log(0) issues
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Categorical Cross-Entropy Loss (Softmax outputs)
def categorical_cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)  # Avoid log(0) issues
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

# Hinge Loss (for SVM)
def hinge_loss(y_true, y_pred):
    return np.mean(np.maximum(0, 1 - y_true * y_pred))

# Example usage
if __name__ == "__main__":
    # Example regression data
    y_true_reg = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred_reg = np.array([2.5, 0.0, 2.0, 8.0])

    print("MSE:", mean_squared_error(y_true_reg, y_pred_reg))
    print("MAE:", mean_absolute_error(y_true_reg, y_pred_reg))
    print("Huber Loss:", huber_loss(y_true_reg, y_pred_reg))

    # Example binary classification data
    y_true_bin = np.array([1, 0, 1, 1])
    y_pred_bin = np.array([0.9, 0.1, 0.8, 0.7])  # Probabilities

    print("Binary Cross-Entropy:", binary_cross_entropy(y_true_bin, y_pred_bin))

    # Example multi-class classification (one-hot encoded labels)
    y_true_cat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    y_pred_cat = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.3, 0.6]])

    print("Categorical Cross-Entropy:", categorical_cross_entropy(y_true_cat, y_pred_cat))

    # Example hinge loss for SVM
    y_true_hinge = np.array([1, -1, 1, -1])
    y_pred_hinge = np.array([0.9, -0.8, 1.2, -1.1])  # SVM decision values

    print("Hinge Loss:", hinge_loss(y_true_hinge, y_pred_hinge))
