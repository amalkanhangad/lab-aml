import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from sklearn.utils import resample

# Load and preprocess data
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
X_train, X_test = X_train.reshape(-1, 784) / 255.0, X_test.reshape(-1, 784) / 255.0

# Select binary classes (0 and 1)
mask_train = (y_train == 0) | (y_train == 1)
mask_test = (y_test == 0) | (y_test == 1)
X_train, y_train = X_train[mask_train], y_train[mask_train]
X_test, y_test = X_test[mask_test], y_test[mask_test]

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

#Training and Predicting Function
def train_and_predict(X_train, y_train, X_test):
    m, c = np.zeros(X_train.shape[1]), 0
    for _ in range(50):
        for i in range(len(X_train)):
            z = np.dot(m, X_train[i]) + c
            m += 0.0001 * X_train[i] * (y_train[i] - sigmoid(z))
            c += 0.0001 * (y_train[i] - sigmoid(z))
    return sigmoid(np.dot(X_test, m) + c)

#Training Multiple Models and Collecting Predictions
all_predictions = []
for _ in range(10):
    X_resampled, y_resampled = resample(X_train, y_train)
    all_predictions.append(train_and_predict(X_resampled, y_resampled, X_test))

all_predictions = np.array(all_predictions)
average_prediction = np.mean(all_predictions, axis=0)

bias = mean_squared_error(y_test, average_prediction)
variance = np.mean(np.var(all_predictions, axis=0))

print("Bias:", bias)
print("Variance:", variance)