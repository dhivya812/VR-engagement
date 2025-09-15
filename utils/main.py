import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, LSTM, Dense, TimeDistributed
from sklearn.model_selection import train_test_split
from utils.preprocess import load_data

# Load dataset
print("📂 Loading dataset...")
X, y = load_data("data/")   # Put your dataset in /data
print("✅ Dataset shape:", X.shape, y.shape)

# Reshape into sequences for LSTM (10-frame sequences)
timesteps = 10
X = X.reshape(-1, timesteps, 48, 48, 1)
y = y[:len(X)]  # Align y with X

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build CNN-LSTM model
model = Sequential([
    TimeDistributed(Conv2D(32, (3,3), activation='relu'), input_shape=(timesteps, 48, 48, 1)),
    TimeDistributed(MaxPooling2D((2,2))),
    TimeDistributed(Flatten()),
    LSTM(64, return_sequences=False),
    Dense(y.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train model
print("🚀 Training model...")
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=16)

# Save model
import os
os.makedirs("models", exist_ok=True)
model.save("models/vr_cnn_lstm.h5")
print("💾 Model saved in models/vr_cnn_lstm.h5")
