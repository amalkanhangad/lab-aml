import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Generate dummy data
num_sequences, sequence_length, num_features, num_classes = 1000, 100, 1, 10
X = np.random.rand(num_sequences, sequence_length, num_features)
y = np.random.randint(0, num_classes, num_sequences)

# Define the 1D CNN model
model = Sequential([
    Conv1D(filters=64, kernal_size=3, activation='relu', input_shape=(sequence_length, num_features)),
    MaxPooling1D(poolsize=2),
    Dropout(0.5),
    Conv1D(128, 3, activation='relu'),
    MaxPooling1D(2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compile and train the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)