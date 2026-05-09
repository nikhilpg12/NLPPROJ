import json
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import random

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    LSTM,
    Bidirectional,
    Conv1D,
    Dense,
    Dropout,
    Embedding,
    MaxPooling1D,
    SpatialDropout1D,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Reproducibility

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


#Load data
train_df = pd.read_csv("train.csv")
eval_df = pd.read_csv("evaluation.csv")
test_df = pd.read_csv("test.csv")

X_train_text = train_df["text"].astype(str)
X_eval_text = eval_df["text"].astype(str)
X_test_text = test_df["text"].astype(str)

y_train_text = train_df["author_id"]
y_eval_text = eval_df["author_id"]
y_test_text = test_df["author_id"]


# Encode labels
label_encoder = LabelEncoder()

y_train = label_encoder.fit_transform(y_train_text)
y_eval = label_encoder.transform(y_eval_text)
y_test = label_encoder.transform(y_test_text)

num_authors = len(label_encoder.classes_)

print("Number of authors:", num_authors)
print("Authors:", list(label_encoder.classes_))


# Tokenise text
vocab_size = 15000
max_length = 300

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train_text)

X_train = tokenizer.texts_to_sequences(X_train_text)
X_eval = tokenizer.texts_to_sequences(X_eval_text)
X_test = tokenizer.texts_to_sequences(X_test_text)

X_train = pad_sequences(X_train, maxlen=max_length, padding="post", truncating="post")
X_eval = pad_sequences(X_eval, maxlen=max_length, padding="post", truncating="post")
X_test = pad_sequences(X_test, maxlen=max_length, padding="post", truncating="post")


# Improved CNN-LSTM Model
input_layer = tf.keras.Input(shape=(max_length,))
x = Embedding(input_dim=vocab_size, output_dim=128)(input_layer)
x = SpatialDropout1D(0.2)(x)
x = Conv1D(filters=64, kernel_size=3, activation="relu", padding="same")(x)
x = MaxPooling1D(pool_size=2)(x)
x = Bidirectional(LSTM(64, dropout=0.3, recurrent_dropout=0.2))(x)
dense_out = Dense(64, activation="relu")(x)
x = Dropout(0.5)(dense_out)
x = Dense(48, activation="relu")(x)
output_layer = Dense(num_authors, activation="softmax")(x)

model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# Train model
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=0.00001
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_eval, y_eval),
    epochs=50,
    batch_size=16,
    callbacks=[early_stop, reduce_lr]
)


# Test evaluation
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\nTest Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=label_encoder.classes_,
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ── Save data for graph.py ──────────────────────────────────────────────────
os.makedirs("out", exist_ok=True)

# 1. Training history
with open("out/history.json", "w") as f:
    json.dump(history.history, f)

# 2. Latent-space activations (dense layer output) for ALL splits
from tensorflow.keras.models import Model

latent_model = Model(inputs=model.input, outputs=dense_out)

latent_rows = []
for X_split, y_split, split_name in [
    (X_train, y_train, "train"),
    (X_eval,  y_eval,  "eval"),
    (X_test,  y_test,  "test"),
]:
    acts = latent_model.predict(X_split, verbose=0)
    for i, vec in enumerate(acts):
        latent_rows.append({
            "split": split_name,
            "true_label": int(y_split[i]),
            "author": label_encoder.classes_[y_split[i]],
            **{f"d{j}": float(v) for j, v in enumerate(vec)},
        })

pd.DataFrame(latent_rows).to_csv("out/latent_activations.csv", index=False)

# 3. Per-sample predictions (test set)
test_pred_probs = model.predict(X_test, verbose=0)
test_pred_labels = np.argmax(test_pred_probs, axis=1)
test_confidence = test_pred_probs.max(axis=1)

pred_df = pd.DataFrame({
    "true_label": y_test,
    "pred_label": test_pred_labels,
    "true_author": label_encoder.inverse_transform(y_test),
    "pred_author": label_encoder.inverse_transform(test_pred_labels),
    "confidence": test_confidence,
    "correct": (y_test == test_pred_labels).astype(int),
})
pred_df.to_csv("out/test_predictions.csv", index=False)

# 4. Confusion matrix + class names
cm = confusion_matrix(y_test, test_pred_labels)
np.save("out/confusion_matrix.npy", cm)
with open("out/class_names.json", "w") as f:
    json.dump(list(label_encoder.classes_), f)

print("\nPlot data saved to out/")

#CNN performs better than CNN-LSTM on smaller datasets because convolutional layers effectively
# capture local stylistic patterns, while LSTM layers introduce additional
#  complexity that requires larger datasets to generalise well.