import os
import json

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    Concatenate,
    Conv1D,
    Dense,
    Dropout,
    Embedding,
    GlobalMaxPooling1D,
    Input,
)
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

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



#Encode labels
label_encoder = LabelEncoder()

y_train = label_encoder.fit_transform(y_train_text)
y_eval = label_encoder.transform(y_eval_text)
y_test = label_encoder.transform(y_test_text)

num_authors = len(label_encoder.classes_)

print("Number of authors:", num_authors)
print("Authors:", list(label_encoder.classes_))



#Tokenise text
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



#Build Multi-Filter CNN Model
input_layer = Input(shape=(max_length,))

embedding = Embedding(
    input_dim=vocab_size,
    output_dim=128
)(input_layer)

conv3 = Conv1D(filters=128, kernel_size=3, activation="relu")(embedding)
conv5 = Conv1D(filters=128, kernel_size=5, activation="relu")(embedding)
conv7 = Conv1D(filters=128, kernel_size=7, activation="relu")(embedding)

pool3 = GlobalMaxPooling1D()(conv3)
pool5 = GlobalMaxPooling1D()(conv5)
pool7 = GlobalMaxPooling1D()(conv7)

concat = Concatenate()([pool3, pool5, pool7])

dense = Dense(64, activation="relu")(concat)
dropout = Dropout(0.5)(dense)

output_layer = Dense(num_authors, activation="softmax")(dropout)

model = Model(inputs=input_layer, outputs=output_layer)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Train model
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_eval, y_eval),
    epochs=50,
    batch_size=16,
    callbacks=[early_stop]
)

# Evaluate model
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

# Create out/ dir
os.makedirs("out", exist_ok=True)

# Training history
with open("out/history.json", "w") as f:
    json.dump(history.history, f)

# Latent-space activations (dense layer output) for ALL splits
latent_model = Model(inputs=model.input, outputs=model.get_layer("dense").output)

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

# Per-sample predictions (test set)
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

# Confusion matrix + class names
cm = confusion_matrix(y_test, test_pred_labels)
np.save("out/confusion_matrix.npy", cm)
with open("out/class_names.json", "w") as f:
    json.dump(list(label_encoder.classes_), f)

print("\nPlot data saved: history.json, latent_activations.csv, test_predictions.csv, confusion_matrix.npy, class_names.json")


#Results:
# #The multi-filter CNN model achieved a test accuracy of 85.5% and a macro F1-score of 0.86. 
# The model demonstrated strong performance across most authors, particularly those with distinctive writing styles. 
# The use of multiple convolutional filters enabled the model to capture stylistic features at different n-gram levels, 
# significantly improving classification performance compared to simpler CNN architectures.

#discussion:
#The results indicate that the multi-filter CNN architecture is highly effective for authorship attribution on this dataset. 
# By using parallel convolutional layers with different kernel sizes, the model was able to capture both short and long stylistic patterns. 
# Some confusion remained between authors with similar writing styles, suggesting overlapping linguistic features. 
# Overall, the model achieved a good balance between generalisation and complexity.


#Increasing architectural diversity to a multi-filter CNN iproved performance more effectively than increasing model depth(CNN-LSTM) on this dataset.