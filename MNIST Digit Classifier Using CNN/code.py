import gzip
import os
import pickle
import urllib.request
from typing import Optional

import numpy as np
from numpy import ndarray

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def softmax(x):
    """Compute softmax values for each set of scores in x (numerically stable)."""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)


def load_mnist(data_dir="/tmp/mnist"):
    """Download (if needed) and load the MNIST dataset using pure Python."""
    os.makedirs(data_dir, exist_ok=True)

    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz",
    }

    def download(filename):
        local_path = os.path.join(data_dir, filename)
        if not os.path.exists(local_path):
            print(f"  Downloading {filename}...")
            urllib.request.urlretrieve(base_url + filename, local_path)
        return local_path

    def read_images(filename):
        with gzip.open(download(filename), "rb") as f:
            raw = np.frombuffer(f.read(), dtype=np.uint8)
        # First 16 bytes are header (magic, count, rows, cols)
        return raw[16:].reshape(-1, 28, 28)

    def read_labels(filename):
        with gzip.open(download(filename), "rb") as f:
            raw = np.frombuffer(f.read(), dtype=np.uint8)
        # First 8 bytes are header (magic, count)
        return raw[8:]

    print("Loading MNIST dataset...")
    X_train = read_images(files["train_images"]).astype(np.float32) / 255.0
    y_train = read_labels(files["train_labels"]).astype(np.int32)
    X_test = read_images(files["test_images"]).astype(np.float32) / 255.0
    y_test = read_labels(files["test_labels"]).astype(np.int32)

    print(f"Data loaded. Training set: {X_train.shape}, Test set: {X_test.shape}")
    return X_train, y_train, X_test, y_test


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """Transform image patches into columns for efficient convolution."""
    input_data = input_data.astype(np.float32, copy=False)
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], "constant")
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w), dtype=np.float32)

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """Transform columns back to image format (reverse of im2col)."""
    N, C, H, W = input_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(
        0, 3, 4, 5, 1, 2
    )

    img = np.zeros((N, C, H + 2 * pad, W + 2 * pad), dtype=np.float32)

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    if pad > 0:
        return img[:, :, pad:-pad, pad:-pad]
    return img


# ---------------------------------------------------------------------------
# Layer classes
# ---------------------------------------------------------------------------


class Convolution:
    def __init__(
        self,
        input_channels,
        output_channels,
        kernel_size=3,
        stride=1,
        pad=1,
        learning_rate=0.01,
    ):
        scale = np.sqrt(1.0 / (input_channels * kernel_size * kernel_size)).astype(
            np.float32
        )
        self.W = scale * np.random.randn(
            output_channels, input_channels, kernel_size, kernel_size
        ).astype(np.float32)
        self.b = np.zeros(output_channels, dtype=np.float32)
        self.stride = stride
        self.pad = pad
        self.lr = learning_rate
        self.x: Optional[ndarray] = None
        self.col: Optional[ndarray] = None
        self.col_W: Optional[ndarray] = None
        self.dW: Optional[ndarray] = None
        self.db: Optional[ndarray] = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = (H + 2 * self.pad - FH) // self.stride + 1
        out_w = (W + 2 * self.pad - FW) // self.stride + 1

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T
        out = np.dot(col, col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W
        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout_reshaped = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        assert self.col is not None and self.col_W is not None and self.x is not None
        self.db = np.sum(dout_reshaped, axis=0)
        self.dW = np.dot(self.col.T, dout_reshaped).T.reshape(FN, C, FH, FW)
        dcol = np.dot(dout_reshaped, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)
        return dx


class MaxPooling:
    def __init__(self, pool_size=2, stride=2, pad=0):
        self.pool_h = pool_size
        self.pool_w = pool_size
        self.stride = stride
        self.pad = pad
        self.x: Optional[ndarray] = None
        self.arg_max: Optional[ndarray] = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = (H + 2 * self.pad - self.pool_h) // self.stride + 1
        out_w = (W + 2 * self.pad - self.pool_w) // self.stride + 1

        col = im2col(
            x.reshape(N * C, 1, H, W), self.pool_h, self.pool_w, self.stride, self.pad
        )
        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1).reshape(N, C, out_h, out_w)

        self.x = x
        self.arg_max = arg_max
        return out

    def backward(self, dout):
        assert self.x is not None and self.arg_max is not None
        N, C, out_h, out_w = dout.shape
        _, _, H_x, W_x = self.x.shape
        pool_size = self.pool_h * self.pool_w
        dmax_size = N * C * out_h * out_w

        dmax = np.zeros((dmax_size, pool_size), dtype=np.float32)
        dmax[np.arange(dmax_size), self.arg_max] = dout.flatten()

        dx = col2im(
            dmax, (N * C, 1, H_x, W_x), self.pool_h, self.pool_w, self.stride, self.pad
        )
        return dx.reshape(self.x.shape)


class Flatten:
    def __init__(self):
        self.original_shape = None

    def forward(self, x):
        self.original_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self.original_shape)


class FullyConnected:
    def __init__(self, input_size, output_size, learning_rate=0.01):
        scale = np.sqrt(2.0 / (input_size + output_size)).astype(np.float32)
        self.W = scale * np.random.randn(input_size, output_size).astype(np.float32)
        self.b = np.zeros(output_size, dtype=np.float32)
        self.lr = learning_rate
        self.x: Optional[ndarray] = None
        self.dW: Optional[ndarray] = None
        self.db: Optional[ndarray] = None

    def forward(self, x):
        x_f32: ndarray = x.astype(np.float32, copy=False)
        self.x = x_f32
        return np.dot(x_f32, self.W) + self.b

    def backward(self, dout):
        assert self.x is not None
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return np.dot(dout, self.W.T)


class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = x <= 0
        out = x.astype(np.float32, copy=True)
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        return dout


class SoftmaxWithLoss:
    def __init__(self):
        self.loss: Optional[float] = None
        self.y: Optional[ndarray] = None  # softmax probabilities
        self.t_one_hot: Optional[ndarray] = None  # one-hot targets

    def forward(self, x, t):
        x = x.astype(np.float32, copy=False)
        self.y = softmax(x)

        # Convert integer labels to one-hot if needed
        if t.ndim == 1 or (t.ndim == 2 and t.shape[1] == 1):
            num_classes = x.shape[1]
            t_flat = t.flatten().astype(np.int32)
            t_one_hot = np.zeros((t_flat.size, num_classes), dtype=np.float32)
            t_one_hot[np.arange(t_flat.size), t_flat] = 1.0
            self.t_one_hot = t_one_hot
        else:
            self.t_one_hot = t.astype(np.float32, copy=False)

        epsilon = 1e-7
        assert self.t_one_hot is not None and self.y is not None
        batch_size = self.t_one_hot.shape[0]
        self.loss = float(
            -np.sum(self.t_one_hot * np.log(self.y + epsilon)) / batch_size
        )
        return self.loss

    def backward(self, dout: float = 1.0):
        assert self.y is not None and self.t_one_hot is not None
        batch_size = self.t_one_hot.shape[0]
        return (self.y - self.t_one_hot) / batch_size * dout


# ---------------------------------------------------------------------------
# CNN model
# ---------------------------------------------------------------------------


class CNN:
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate

        # Architecture: Conv(32) -> Pool -> Conv(64) -> Pool -> FC(128) -> FC(10)
        self.conv1 = Convolution(1, 32, 3, 1, 1, self.lr)
        self.relu1 = ReLU()
        self.pool1 = MaxPooling(2, 2)
        self.conv2 = Convolution(32, 64, 3, 1, 1, self.lr)
        self.relu2 = ReLU()
        self.pool2 = MaxPooling(2, 2)
        self.flatten = Flatten()
        self.fc1 = FullyConnected(64 * 7 * 7, 128, self.lr)
        self.relu3 = ReLU()
        self.fc2 = FullyConnected(128, 10, self.lr)
        self.softmax = SoftmaxWithLoss()

        self.layers = [
            self.conv1,
            self.relu1,
            self.pool1,
            self.conv2,
            self.relu2,
            self.pool2,
            self.flatten,
            self.fc1,
            self.relu3,
            self.fc2,
        ]
        self.params = [self.conv1, self.conv2, self.fc1, self.fc2]

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def predict(self, x):
        """Run a forward pass and return raw logits."""
        if x.ndim == 3:
            x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
        elif x.ndim == 2:
            x = x.reshape(1, 1, x.shape[0], x.shape[1])
        x = x.astype(np.float32, copy=False)

        h = x
        for layer in self.layers:
            h = layer.forward(h)
        return h

    def loss(self, x, t):
        """Forward pass + cross-entropy loss."""
        y = self.predict(x)
        return self.softmax.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        """Compute classification accuracy (supports batched evaluation)."""
        n_data = x.shape[0]
        if x.ndim == 3:
            x = x.reshape(n_data, 1, x.shape[1], x.shape[2])
        x = x.astype(np.float32, copy=False)
        if t.ndim == 2:
            t = np.argmax(t, axis=1)
        t = t.astype(np.int32)

        correct = 0
        for i in range(0, n_data, batch_size):
            x_b = x[i : i + batch_size]
            t_b = t[i : i + batch_size]
            y_b = self.predict(x_b)
            correct += np.sum(np.argmax(y_b, axis=1) == t_b)
        return correct / n_data

    def gradient(self, x, t):
        """Compute gradients via backpropagation."""
        if x.ndim == 3:
            x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
        x = x.astype(np.float32, copy=False)

        self.loss(x, t)  # forward
        dout = self.softmax.backward(1.0)  # backward through loss
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

    def update_params(self):
        """SGD parameter update."""
        for layer in self.params:
            layer.W -= self.lr * layer.dW.astype(np.float32, copy=False)
            layer.b -= self.lr * layer.db.astype(np.float32, copy=False)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, x_train, t_train, x_val, t_val, epochs=5, batch_size=100):
        """Train the model and return per-iteration loss and per-epoch accuracy."""
        train_size = x_train.shape[0]
        iter_per_epoch = max(train_size // batch_size, 1)
        train_loss_list, train_acc_list, val_acc_list = [], [], []

        print(f"Starting training for {epochs} epochs...")
        for epoch in range(epochs):
            idx = np.random.permutation(train_size)
            x_train_shuffled = x_train[idx]
            t_train_shuffled = t_train[idx]
            epoch_loss = 0.0

            for i in range(iter_per_epoch):
                start = i * batch_size
                end = start + batch_size
                x_batch = x_train_shuffled[start:end]
                t_batch = t_train_shuffled[start:end]

                self.gradient(x_batch, t_batch)
                self.update_params()

                loss = self.softmax.loss
                assert loss is not None
                train_loss_list.append(loss)
                epoch_loss += loss

                if (i + 1) % 100 == 0:
                    print(
                        f"  Epoch {epoch + 1}, "
                        f"Iteration {i + 1}/{iter_per_epoch}, "
                        f"Batch Loss: {loss:.4f}"
                    )

            print(f"Epoch {epoch + 1}/{epochs} finished. Evaluating...")
            eval_size = 1000
            train_acc = self.accuracy(
                x_train[:eval_size], t_train[:eval_size], batch_size=500
            )
            val_acc = self.accuracy(
                x_val[:eval_size], t_val[:eval_size], batch_size=500
            )
            train_acc_list.append(train_acc)
            val_acc_list.append(val_acc)

            print(
                f"  Avg Loss: {epoch_loss / iter_per_epoch:.4f}, "
                f"Train Acc (on {eval_size}): {train_acc:.4f}, "
                f"Val Acc (on {eval_size}): {val_acc:.4f}"
            )
            print("-" * 30)

        return train_loss_list, train_acc_list, val_acc_list

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self, file_path):
        """Serialise model weights to a pickle file."""
        params = {
            "conv1_W": self.conv1.W,
            "conv1_b": self.conv1.b,
            "conv2_W": self.conv2.W,
            "conv2_b": self.conv2.b,
            "fc1_W": self.fc1.W,
            "fc1_b": self.fc1.b,
            "fc2_W": self.fc2.W,
            "fc2_b": self.fc2.b,
            "lr": self.lr,
        }
        with open(file_path, "wb") as f:
            pickle.dump(params, f)
        print(f"Model parameters saved to {file_path}")

    @staticmethod
    def load_model(file_path, learning_rate=0.01):
        """Load model weights from a pickle file."""
        with open(file_path, "rb") as f:
            params = pickle.load(f)
        lr = params.get("lr", learning_rate)
        model = CNN(learning_rate=lr)
        model.conv1.W = params["conv1_W"]
        model.conv1.b = params["conv1_b"]
        model.conv2.W = params["conv2_W"]
        model.conv2.b = params["conv2_b"]
        model.fc1.W = params["fc1_W"]
        model.fc1.b = params["fc1_b"]
        model.fc2.W = params["fc2_W"]
        model.fc2.b = params["fc2_b"]
        print(f"Model parameters loaded from {file_path}")
        return model


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_mnist()

    model = CNN(learning_rate=0.01)
    print("Training CNN...")
    train_loss, train_acc, val_acc = model.train(
        X_train, y_train, X_test, y_test, epochs=3, batch_size=100
    )

    print("Evaluating on full test set...")
    test_acc = model.accuracy(X_test, y_test, batch_size=500)
    print(f"Test accuracy: {test_acc:.4f}")

    model.save_model("mnist_cnn_params.pkl")

    # Optional: reload and re-evaluate
    # loaded_model = CNN.load_model('mnist_cnn_params.pkl')
    # test_acc_loaded = loaded_model.accuracy(X_test, y_test, batch_size=500)
    # print(f"Loaded model test accuracy: {test_acc_loaded:.4f}")
