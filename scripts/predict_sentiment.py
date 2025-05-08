# -*- coding: utf-8 -*-

import os
import numpy as np
import pickle
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from preprocess import preprocess_text

# === CONFIG ===
MAX_WORDS = 10000
MAX_LEN = 200

# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "models", "best_model.h5")
tokenizer_path = os.path.join(script_dir, "split_data", "tokenizer.pkl")

# Load tokenizer
with open(tokenizer_path, "rb") as f:
    tokenizer = pickle.load(f)
tokenizer.num_words = MAX_WORDS


# Load trained model
model = load_model(model_path)

# Preprocessing function
def text_processing(texts):
    cleaned = [preprocess_text(t) for t in texts]
    sequences = tokenizer.texts_to_sequences(cleaned)
    # Replace words with index >= MAX_WORDS with 1 (usually <OOV> token)
    clipped_sequences = [[i if i < MAX_WORDS else 1 for i in seq] for seq in sequences]
    padded = pad_sequences(clipped_sequences, maxlen=MAX_LEN)
    return padded


# Predict function
def predict_sentiment(texts):
    processed = text_processing(texts)
    predictions = model.predict(processed)
    results = ["Positive" if p > 0.7 else "Negative" for p in predictions]
    return zip(texts, results, predictions.flatten())

if __name__ == "__main__":
    sample_texts = [
        "This movie was absolutely amazing and inspiring!",
        "I hated this movie. It was boring and too long.",
        "An average film with good acting but weak plot.",
        "Its Pugh who carries the film, making us care about her jaded world-weariness but always keeping it funny. Which is not to say that there is no other good acting in Thunderbolts.",
        "Thunderbolts succeeds in spite of Marvel’s built in hurdles and its uneven script. The film remembers that our love of superheroes doesn’t stem from what these overpowered beings can do but what these humans who become icons overcame to earn the title."
    ]
    
    outputs = predict_sentiment(sample_texts)
    for text, label, score in outputs:
        print("\nText: {}\nPredicted Sentiment: {} (Score: {:.4f})".format(text, label, score))
