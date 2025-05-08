# -*- coding: utf-8 -*-

import numpy as np
import os
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Embedding, Bidirectional, LSTM, GlobalMaxPooling1D, Dense, Dropout, BatchNormalization

# === CONFIG ===
MAX_WORDS = 10000
MAX_LEN = 200
EMBEDDING_DIM = 128
EPOCHS = 10
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.2

# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "split_data")
model_dir = os.path.join(script_dir, "models")
best_model_path = os.path.join(model_dir, "best_model.h5")

# Create model directory if it doesn't exist
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Load data
X_train = np.load(os.path.join(data_path, "X_train.npy"))
y_train = np.load(os.path.join(data_path, "y_train.npy"))

# Build model
# model = Sequential([
#     Embedding(input_dim=MAX_WORDS, output_dim=EMBEDDING_DIM, input_length=MAX_LEN),
#     LSTM(128),
#     Dropout(0.5),
#     Dense(64, activation='relu'),
#     Dropout(0.5),
#     Dense(1, activation='sigmoid')
# ])

model = Sequential([
    Embedding(input_dim=MAX_WORDS, output_dim=EMBEDDING_DIM, input_length=MAX_LEN),
    
    # Bidirectional + stacked LSTM
    Bidirectional(LSTM(128, return_sequences=True)),
    Dropout(0.3),
    BatchNormalization(),
    
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.3),
    
    # Global max pooling instead of flattening LSTM outputs
    GlobalMaxPooling1D(),
    
    Dense(64, activation='relu'),
    Dropout(0.5),
    
    Dense(1, activation='sigmoid')
])

print(model.summary())

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=2, verbose=1)
checkpoint = ModelCheckpoint(best_model_path, monitor='val_loss', save_best_only=True, verbose=1)

# Train
model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=64,
    callbacks=[early_stop, checkpoint]
)

print("Training complete. Best model saved at:", best_model_path)
