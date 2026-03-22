import os
import urllib.request

import numpy as np

# ─────────────────────────────────────────────
# 1. Download & Load Data
# ─────────────────────────────────────────────


def download_mnist(path="mnist.npz"):
    """Download the MNIST dataset if it is not already present."""
    if not os.path.exists(path):
        print("Downloading mnist.npz ...")
        url = "https://s3.amazonaws.com/img-datasets/mnist.npz"
        urllib.request.urlretrieve(url, path)
        print("Download complete.")
    else:
        print("mnist.npz already exists, skipping download.")


def load_mnist(path="mnist.npz"):
    """Load MNIST from an .npz file and return raw splits."""
    data = np.load(path)
    x_train = data["x_train"]
    y_train = data["y_train"]
    x_test = data["x_test"]
    y_test = data["y_test"]
    return x_train, y_train, x_test, y_test


# ─────────────────────────────────────────────
# 2. Preprocessing
# ─────────────────────────────────────────────


def preprocess(x_train, y_train, x_test, y_test, num_classes=10):
    """
    Flatten 28x28 images → 784-dim vectors, normalise to [0, 1],
    and one-hot encode the labels.
    """
    # Flatten and normalise pixel values
    x_train = x_train.reshape(-1, 28 * 28).astype(np.float32) / 255.0
    x_test = x_test.reshape(-1, 28 * 28).astype(np.float32) / 255.0

    # One-hot encode labels using an identity matrix as a lookup table
    y_train_oh = np.eye(num_classes)[y_train]
    y_test_oh = np.eye(num_classes)[y_test]

    return x_train, y_train_oh, x_test, y_test_oh


# ─────────────────────────────────────────────
# 3. Activation Functions
# ─────────────────────────────────────────────


def relu(x):
    """Rectified Linear Unit: f(x) = max(0, x)."""
    return np.maximum(0, x)


def relu_derivative(x):
    """Derivative of ReLU: 1 where x > 0, else 0."""
    return (x > 0).astype(np.float32)


def softmax(x):
    """
    Numerically stable softmax along axis=1.
    Subtracts the row-wise maximum before exponentiating to avoid overflow.
    """
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)


# ─────────────────────────────────────────────
# 4. Weight Initialisation
# ─────────────────────────────────────────────


def init_weights(input_size, hidden_size, output_size):
    """
    He (Kaiming) initialisation for weights – recommended for ReLU networks.
    Biases are initialised to zero.
    """
    W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
    b2 = np.zeros((1, output_size))
    return W1, b1, W2, b2


# ─────────────────────────────────────────────
# 5. Forward Pass
# ─────────────────────────────────────────────


def forward(x, W1, b1, W2, b2):
    """
    Two-layer forward pass:
      Input  → Linear → ReLU → Linear → Softmax
    Returns all intermediate values needed for backprop.
    """
    z1 = np.dot(x, W1) + b1  # (batch, hidden)
    a1 = relu(z1)  # (batch, hidden)
    z2 = np.dot(a1, W2) + b2  # (batch, output)
    a2 = softmax(z2)  # (batch, output)  – probabilities
    return z1, a1, z2, a2


# ─────────────────────────────────────────────
# 6. Loss
# ─────────────────────────────────────────────


def cross_entropy_loss(y_pred, y_true, batch_size):
    """
    Mean categorical cross-entropy loss.
    A small epsilon (1e-8) is added to y_pred to avoid log(0).
    """
    return -np.sum(y_true * np.log(y_pred + 1e-8)) / batch_size


# ─────────────────────────────────────────────
# 7. Backward Pass
# ─────────────────────────────────────────────


def backward(x_batch, y_batch, z1, a1, a2, W1, W2, batch_size):
    """
    Compute gradients of all weights and biases via backpropagation.

    Derivations:
      dL/dz2 = a2 - y  (combined softmax + cross-entropy gradient)
      dL/dW2 = a1^T · dz2
      dL/db2 = sum(dz2)
      dL/da1 = dz2 · W2^T
      dL/dz1 = da1 * relu'(z1)
      dL/dW1 = x^T · dz1
      dL/db1 = sum(dz1)
    """
    dz2 = a2 - y_batch  # (batch, output)
    dW2 = np.dot(a1.T, dz2) / batch_size  # (hidden, output)
    db2 = np.sum(dz2, axis=0, keepdims=True) / batch_size  # (1, output)

    da1 = np.dot(dz2, W2.T)  # (batch, hidden)
    dz1 = da1 * relu_derivative(z1)  # (batch, hidden)
    dW1 = np.dot(x_batch.T, dz1) / batch_size  # (input, hidden)
    db1 = np.sum(dz1, axis=0, keepdims=True) / batch_size  # (1, hidden)

    return dW1, db1, dW2, db2


# ─────────────────────────────────────────────
# 8. Training
# ─────────────────────────────────────────────


def train(x_train, y_train_oh, W1, b1, W2, b2, epochs, batch_size, learning_rate):
    """
    Mini-batch stochastic gradient descent training loop.
    The dataset is shuffled at the beginning of every epoch.
    """
    num_samples = x_train.shape[0]
    num_batches = num_samples // batch_size

    for epoch in range(epochs):
        # Shuffle training data
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        x_train = x_train[indices]
        y_train_oh = y_train_oh[indices]

        epoch_loss = 0.0

        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size
            x_batch = x_train[start:end]
            y_batch = y_train_oh[start:end]

            # Forward pass
            z1, a1, z2, a2 = forward(x_batch, W1, b1, W2, b2)

            # Loss
            loss = cross_entropy_loss(a2, y_batch, batch_size)
            epoch_loss += loss

            # Backward pass
            dW1, db1, dW2, db2 = backward(
                x_batch, y_batch, z1, a1, a2, W1, W2, batch_size
            )

            # Gradient descent update
            W1 -= learning_rate * dW1
            b1 -= learning_rate * db1
            W2 -= learning_rate * dW2
            b2 -= learning_rate * db2

        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch + 1}/{epochs}  |  Loss: {avg_loss:.4f}")

    return W1, b1, W2, b2


# ─────────────────────────────────────────────
# 9. Evaluation
# ─────────────────────────────────────────────


def evaluate(x_test, y_test, W1, b1, W2, b2):
    """Run a single forward pass on the test set and report accuracy."""
    _, _, _, a2_test = forward(x_test, W1, b1, W2, b2)
    predictions = np.argmax(a2_test, axis=1)
    accuracy = np.mean(predictions == y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    return accuracy


# ─────────────────────────────────────────────
# 10. Main
# ─────────────────────────────────────────────


def main():
    # ── Hyperparameters ──────────────────────
    input_size = 28 * 28  # 784
    hidden_size = 128
    output_size = 10  # digits 0-9
    learning_rate = 0.1
    epochs = 10
    batch_size = 128
    # ─────────────────────────────────────────

    # Download and load
    download_mnist()
    x_train_raw, y_train_raw, x_test_raw, y_test_raw = load_mnist()

    # Preprocess
    x_train, y_train_oh, x_test, y_test_oh = preprocess(
        x_train_raw, y_train_raw, x_test_raw, y_test_raw
    )

    print(f"\nTraining samples : {x_train.shape[0]}")
    print(f"Test samples     : {x_test.shape[0]}")
    print(f"Input features   : {x_train.shape[1]}")
    print(f"Classes          : {output_size}\n")

    # Initialise weights
    W1, b1, W2, b2 = init_weights(input_size, hidden_size, output_size)

    # Train
    W1, b1, W2, b2 = train(
        x_train, y_train_oh, W1, b1, W2, b2, epochs, batch_size, learning_rate
    )

    # Evaluate – pass raw integer labels for argmax comparison
    print()
    evaluate(x_test, y_test_raw, W1, b1, W2, b2)


if __name__ == "__main__":
    main()
