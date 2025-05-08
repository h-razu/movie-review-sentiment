# -*- coding: utf-8 -*-

import os
import numpy as np
import pickle
import pandas as pd
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, GlobalMaxPooling1D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK

# Constants
MAX_WORDS = 10000
MAX_LEN = 200

# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "split_data")
model_dir = os.path.join(script_dir, "models")
tuning_log_dir = os.path.join(script_dir, "tuner_logs")
best_model_path = os.path.join(model_dir, "best_tuned_model.h5")
results_csv_path = os.path.join(model_dir, "tuning_results.csv")

# Create model directory if it doesn't exist
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Load data
X_train = np.load(os.path.join(data_path, "X_train.npy"))
y_train = np.load(os.path.join(data_path, "y_train.npy"))

# Build model function
def build_model(params):
    model = Sequential()
    model.add(
        Embedding(
            input_dim=MAX_WORDS,
            output_dim=params['embedding_dim'],
            input_length=MAX_LEN
        )
    )

    model.add(
        Bidirectional(
            LSTM(
                units=params['lstm_units'],
                return_sequences=True
            )
        )
    )

    model.add(Dropout(params['dropout_1']))
    model.add(GlobalMaxPooling1D())
    model.add(
        Dense(
            units=params['dense_units'],
            activation="relu"
        )
    )
    model.add(Dropout(params['dropout_2']))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=Adam(lr=params['lr']),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

# Objective function for Hyperopt
def objective(params):
    # Build the model
    model = build_model(params)

    # Early stopping
    early_stop = EarlyStopping(monitor="val_loss", patience=2, verbose=1)

    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=5,
        validation_split=0.2,
        batch_size=64,
        callbacks=[early_stop],
        verbose=0
    )

    # Get validation accuracy
    val_accuracy = history.history['val_accuracy'][-1]

    # Return the result in a format Hyperopt expects
    return {
        'loss': -val_accuracy,  # Hyperopt minimizes the loss, so use negative of val_accuracy
        'status': STATUS_OK,
        'params': params  # Return the parameters for tracking purposes
    }

# Hyperparameter space
space = {
    'embedding_dim': hp.choice('embedding_dim', [128, 256]),
    'lstm_units': hp.choice('lstm_units', [64, 128, 256]),
    'dropout_1': hp.uniform('dropout_1', 0.3, 0.6),
    'dropout_2': hp.uniform('dropout_2', 0.3, 0.6),
    'dense_units': hp.choice('dense_units', [64, 128]),
    'lr': hp.loguniform('lr', np.log(1e-4), np.log(1e-2))
}

# Set up Trials object to keep track of results
trials = Trials()

# Run the optimization process
best = fmin(
    fn=objective, 
    space=space, 
    algo=tpe.suggest, 
    max_evals=10,  # Set number of trials
    trials=trials
)

# Print the best hyperparameters
print("Best Hyperparameters:")
for param, value in best.items():
    print("{}: {}".format(param, value))

# Rebuild model with best parameters and retrain
best_model = build_model(best)
best_model.fit(
    X_train, y_train,
    epochs=5,
    validation_split=0.2,
    batch_size=64
)
# Save best model
best_model.save(best_model_path)
print("Best model saved at:", best_model_path)

# Save tuning results
results = []
for trial in trials.trials:
    trial_data = trial['result']['params']
    trial_data['val_accuracy'] = -trial['result']['loss']  # Reverse the sign for accuracy
    results.append(trial_data)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="val_accuracy", ascending=False)
results_df.to_csv(results_csv_path, index=False)

print("All tuning results saved to: {}".format(results_csv_path))
print(results_df.head())
