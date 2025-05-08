# -*- coding: utf-8 -*-

import numpy as np
import os
from keras.models import load_model

# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "split_data")
model_dir = os.path.join(script_dir, "models")
best_model_path = os.path.join(model_dir, "best_model.h5")

# Load test data
X_test = np.load(os.path.join(data_path, "X_test.npy"))
y_test = np.load(os.path.join(data_path, "y_test.npy"))

# Load best saved model
model = load_model(best_model_path)

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print("Test Accuracy: {:.4f}".format(acc))
