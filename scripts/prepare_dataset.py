# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Constants
MAX_WORDS = 10000
MAX_LEN = 200

# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the script
data_path = os.path.join(script_dir, "clean_imdb_dataset.csv")
output_path = os.path.join(script_dir, "split_data")

# Load cleaned data
df = pd.read_csv(data_path)

# Tokenize text
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(df['clean_review'])
sequences = tokenizer.texts_to_sequences(df['clean_review'])
padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')

# Encode sentiment labels
labels = df["sentiment"].map({"positive": 1, "negative": 0}).values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    padded_sequences, labels, test_size=0.2, random_state=42)

# Save
np.save(os.path.join(output_path, "X_train.npy"), X_train)
np.save(os.path.join(output_path, "X_test.npy"), X_test)
np.save(os.path.join(output_path, "y_train.npy"), y_train)
np.save(os.path.join(output_path, "y_test.npy"), y_test)

# Optionally save tokenizer
import pickle
with open(os.path.join(output_path, "tokenizer.pkl"), "wb") as f:
    pickle.dump(tokenizer, f)

print("✅ Dataset tokenized and saved!")
