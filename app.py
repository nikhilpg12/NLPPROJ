"""
Afrikaans Authorship Attribution Streamlit App
==============================================

Place this file in the same folder as:
    train.csv
    evaluation.csv
    test.csv
    local_mbert/                         optional, for mBERT
    mbert_afrikaans_authorship_model/    optional, if already trained

Run:
    streamlit run app.py

This app aligns with the proposal:
- preprocessing / cleaning / length filtering
- TF-IDF + SVM baseline
- CNN-LSTM neural model
- multilingual BERT model
- Accuracy, Precision, Recall, F1-score, confusion matrix
- robustness by text length
- LIME and SHAP explainability
"""

import os

# Important: avoids Streamlit trying to inspect optional transformer vision modules
# that require torchvision. This fixes the repeated "No module named torchvision" logs.
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import re
import html
import random
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import LabelEncoder

# =========================
# FIXED PARAMETERS
# Change values here, not in the Streamlit app.
# =========================
SEED = 42

TRAIN_FILE = "train.csv"
EVAL_FILE = "evaluation.csv"
TEST_FILE = "test.csv"
FINAL_DATASET_FILE = "final_dataset.csv"

TEXT_COLUMN = "text"
LABEL_COLUMN = "author"      # change to "author_id" only if your CSV uses author_id

MIN_WORDS = 5
MAX_WORDS = 1000
BALANCE_TRAINING_CLASSES = False
REMOVE_DUPLICATES = True

RUN_TFIDF_SVM = True
RUN_CNN_LSTM = True
RUN_MBERT = True            # set True if you want to train/use mBERT in the app

TFIDF_WORD_MAX_FEATURES = 15000
TFIDF_CHAR_MAX_FEATURES = 30000
TFIDF_MAX_ITER = 80

CNN_MAX_LENGTH = 300
CNN_VOCAB_SIZE = 15000
CNN_EPOCHS = 12
CNN_BATCH_SIZE = 16

BERT_LOCAL_DIR = "local_mbert"
BERT_MAX_LENGTH = 256
BERT_EPOCHS = 3
BERT_BATCH_SIZE = 4
BERT_LEARNING_RATE = 1e-5

LIME_NUM_FEATURES = 12
SHAP_MAX_BACKGROUND = 50

OUTPUT_DIR = Path("streamlit_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)

# =========================
# Optional packages
# =========================
try:
    from lime.lime_text import LimeTextExplainer
    LIME_AVAILABLE = True
except Exception:
    LIME_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, Dropout, SpatialDropout1D
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.preprocessing.text import Tokenizer
    TF_AVAILABLE = True
    tf.random.set_seed(SEED)
except Exception:
    TF_AVAILABLE = False

try:
    import torch
    from datasets import Dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        DataCollatorWithPadding,
        TrainingArguments,
        Trainer,
        EarlyStoppingCallback,
        set_seed,
    )
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

# =========================
# Streamlit setup
# =========================
st.set_page_config(page_title="Afrikaans Authorship Attribution", page_icon="✍️", layout="wide")
st.title("Afrikaans Authorship Attribution")
st.caption("TF-IDF + SVM, CNN-LSTM, optional mBERT, with LIME and SHAP explainability")

# =========================
# Data preprocessing
# =========================
def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text)
    text = html.unescape(text)
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")

    # Remove URLs, emails and common web artefacts
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)

    # Remove obvious captions/photo/source lines, as required for clean article text
    caption_patterns = [
        r"\b(foto|photo|image|caption|bron|source|illustrasie|illustration)\s*[:\-].*?(?=\.|$)",
        r"\b(beeld|prent|figuur)\s*[:\-].*?(?=\.|$)",
    ]
    for pattern in caption_patterns:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)

    # Preserve Afrikaans characters and punctuation useful for writing style
    text = re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿ0-9\s\.,;:!?\'\"\-–—()\[\]]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def word_count(text):
    return len(str(text).split())


def length_bucket(n):
    if n <= 50:
        return "short"
    if n <= 150:
        return "medium"
    return "long"


def normalise_columns(df):
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def find_existing_file(names):
    for name in names:
        if Path(name).exists():
            return name
    return None


def load_data_from_same_directory():
    train_path = find_existing_file([TRAIN_FILE, "train(11).csv", "train (11).csv"])
    eval_path = find_existing_file([EVAL_FILE, "evaluation(1).csv", "evaluation (1).csv"])
    test_path = find_existing_file([TEST_FILE, "test(9).csv", "test (9).csv"])

    if train_path and eval_path and test_path:
        train_df = pd.read_csv(train_path)
        eval_df = pd.read_csv(eval_path)
        test_df = pd.read_csv(test_path)
        return train_df, eval_df, test_df

    final_path = find_existing_file([FINAL_DATASET_FILE, "final_dataset(1).csv", "final_dataset (1).csv"])
    if final_path is None:
        raise FileNotFoundError(
            "Could not find train.csv, evaluation.csv and test.csv in the same folder as app.py."
        )

    from sklearn.model_selection import train_test_split
    full_df = normalise_columns(pd.read_csv(final_path))
    if LABEL_COLUMN not in full_df.columns:
        raise ValueError(f"{final_path} must contain the label column: {LABEL_COLUMN}")

    train_df, temp_df = train_test_split(
        full_df,
        test_size=0.30,
        random_state=SEED,
        stratify=full_df[LABEL_COLUMN],
    )
    eval_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=SEED,
        stratify=temp_df[LABEL_COLUMN],
    )
    return train_df, eval_df, test_df


def preprocess_dataframe(df, split_name):
    df = normalise_columns(df)

    if TEXT_COLUMN not in df.columns:
        raise ValueError(f"{split_name}.csv must contain a '{TEXT_COLUMN}' column.")
    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"{split_name}.csv must contain a '{LABEL_COLUMN}' column. Change LABEL_COLUMN at the top of app.py if needed.")

    keep_cols = [TEXT_COLUMN, LABEL_COLUMN]
    for col in ["title", "edition", "language_code"]:
        if col in df.columns:
            keep_cols.append(col)

    df = df[keep_cols].copy()
    before = len(df)

    df = df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN])
    df[TEXT_COLUMN] = df[TEXT_COLUMN].apply(clean_text)
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(str).str.strip()
    df = df[(df[TEXT_COLUMN] != "") & (df[LABEL_COLUMN] != "")]
    df = df[(df[TEXT_COLUMN].str.lower() != "nan") & (df[LABEL_COLUMN].str.lower() != "nan")]

    if REMOVE_DUPLICATES:
        df = df.drop_duplicates(subset=[TEXT_COLUMN])

    df["word_count"] = df[TEXT_COLUMN].apply(word_count)
    df = df[(df["word_count"] >= MIN_WORDS) & (df["word_count"] <= MAX_WORDS)].copy()
    df["length_bucket"] = df["word_count"].apply(length_bucket)

    removed = before - len(df)
    df.attrs["removed_rows"] = removed
    return df.reset_index(drop=True)


def balance_training_data(df):
    min_count = df[LABEL_COLUMN].value_counts().min()
    return (
        df.groupby(LABEL_COLUMN, group_keys=False)
        .apply(lambda x: x.sample(min_count, random_state=SEED))
        .sample(frac=1, random_state=SEED)
        .reset_index(drop=True)
    )

# =========================
# Models
# =========================
@st.cache_resource(show_spinner=False)
def train_tfidf_svm(train_texts, train_labels):
    word_tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=TFIDF_WORD_MAX_FEATURES,
        lowercase=True,
        sublinear_tf=True,
    )
    char_tfidf = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        max_features=TFIDF_CHAR_MAX_FEATURES,
        lowercase=True,
        sublinear_tf=True,
    )
    features = FeatureUnion([
        ("word_tfidf", word_tfidf),
        ("char_tfidf", char_tfidf),
    ])
    model = Pipeline([
        ("features", features),
        ("clf", SGDClassifier(
            loss="log_loss",
            max_iter=TFIDF_MAX_ITER,
            tol=1e-4,
            alpha=1e-5,
            random_state=SEED,
            class_weight="balanced",
        )),
    ])
    model.fit(list(train_texts), list(train_labels))
    return model


@st.cache_resource(show_spinner=False)
def train_cnn_lstm(train_texts, train_labels, eval_texts, eval_labels):
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is not installed. Run: pip install tensorflow")

    le = LabelEncoder()
    y_train = le.fit_transform(list(train_labels))
    y_eval = le.transform(list(eval_labels))

    tokenizer = Tokenizer(num_words=CNN_VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(list(train_texts))

    X_train = pad_sequences(
        tokenizer.texts_to_sequences(list(train_texts)),
        maxlen=CNN_MAX_LENGTH,
        padding="post",
        truncating="post",
    )
    X_eval = pad_sequences(
        tokenizer.texts_to_sequences(list(eval_texts)),
        maxlen=CNN_MAX_LENGTH,
        padding="post",
        truncating="post",
    )

    model = Sequential([
        Embedding(input_dim=CNN_VOCAB_SIZE, output_dim=128),
        SpatialDropout1D(0.20),
        Conv1D(filters=64, kernel_size=3, activation="relu", padding="same"),
        MaxPooling1D(pool_size=2),
        Bidirectional(LSTM(64, dropout=0.30, recurrent_dropout=0.20)),
        Dense(64, activation="relu"),
        Dropout(0.50),
        Dense(len(le.classes_), activation="softmax"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_eval, y_eval),
        epochs=CNN_EPOCHS,
        batch_size=CNN_BATCH_SIZE,
        callbacks=callbacks,
        verbose=0,
    )
    return model, tokenizer, le, history.history


def cnn_predict_proba(texts, model, tokenizer):
    X = pad_sequences(
        tokenizer.texts_to_sequences(list(texts)),
        maxlen=CNN_MAX_LENGTH,
        padding="post",
        truncating="post",
    )
    return model.predict(X, verbose=0)


def transformer_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    return {"accuracy": accuracy_score(labels, preds), "macro_precision": p, "macro_recall": r, "macro_f1": f1}


def training_args(output_dir):
    common = dict(
        output_dir=output_dir,
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=BERT_LEARNING_RATE,
        per_device_train_batch_size=BERT_BATCH_SIZE,
        per_device_eval_batch_size=BERT_BATCH_SIZE,
        num_train_epochs=BERT_EPOCHS,
        weight_decay=0.01,
        warmup_ratio=0.1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        report_to="none",
        seed=SEED,
        fp16=torch.cuda.is_available() if TRANSFORMERS_AVAILABLE else False,
    )
    try:
        return TrainingArguments(eval_strategy="epoch", **common)
    except TypeError:
        return TrainingArguments(evaluation_strategy="epoch", **common)


@st.cache_resource(show_spinner=False)
def train_mbert(train_df, eval_df):
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers, datasets and torch are not installed.")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    set_seed(SEED)

    le = LabelEncoder()
    train_df = train_df.copy()
    eval_df = eval_df.copy()
    train_df["labels"] = le.fit_transform(train_df[LABEL_COLUMN])
    eval_df["labels"] = le.transform(eval_df[LABEL_COLUMN])

    id2label = {i: label for i, label in enumerate(le.classes_)}
    label2id = {label: i for i, label in id2label.items()}

    if not Path(BERT_LOCAL_DIR).exists():
        raise FileNotFoundError(f"mBERT local folder '{BERT_LOCAL_DIR}' was not found. Keep RUN_MBERT=False or add local_mbert/.")

    tokenizer = AutoTokenizer.from_pretrained(BERT_LOCAL_DIR, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        BERT_LOCAL_DIR,
        num_labels=len(le.classes_),
        id2label=id2label,
        label2id=label2id,
        local_files_only=True,
        ignore_mismatched_sizes=True,
    )

    def make_ds(df):
        ds = Dataset.from_pandas(df[[TEXT_COLUMN, "labels"]], preserve_index=False)
        def tok(batch):
            return tokenizer(batch[TEXT_COLUMN], truncation=True, max_length=BERT_MAX_LENGTH)
        ds = ds.map(tok, batched=True)
        return ds.remove_columns([TEXT_COLUMN])

    trainer = Trainer(
        model=model,
        args=training_args(str(OUTPUT_DIR / "mbert_output")),
        train_dataset=make_ds(train_df),
        eval_dataset=make_ds(eval_df),
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=transformer_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()
    return trainer, tokenizer, le


def mbert_predict_proba(texts, trainer, tokenizer):
    ds = Dataset.from_dict({TEXT_COLUMN: list(texts)})
    def tok(batch):
        return tokenizer(batch[TEXT_COLUMN], truncation=True, max_length=BERT_MAX_LENGTH)
    ds = ds.map(tok, batched=True)
    ds = ds.remove_columns([TEXT_COLUMN])
    logits = trainer.predict(ds).predictions
    exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)

# =========================
# Evaluation and plots
# =========================
def evaluate_model(y_true, y_pred, class_names):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0, output_dict=True)
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    return acc, p, r, f1, report, cm


def plot_cm(cm, class_names, title):
    fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 0.8), max(4, len(class_names) * 0.6)))
    im = ax.imshow(cm)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


def length_robustness(df, y_pred):
    tmp = df.copy()
    tmp["predicted_author"] = y_pred
    tmp["correct"] = tmp[LABEL_COLUMN].astype(str) == tmp["predicted_author"].astype(str)
    rows = []
    for bucket, group in tmp.groupby("length_bucket"):
        rows.append({
            "length_bucket": bucket,
            "samples": len(group),
            "accuracy": group["correct"].mean(),
            "average_words": group["word_count"].mean(),
        })
    return pd.DataFrame(rows)

# =========================
# Explainability
# =========================
def show_lime(text, predict_fn, class_names):
    if not LIME_AVAILABLE:
        st.error("LIME is not installed. Run: pip install lime")
        return
    explainer = LimeTextExplainer(class_names=list(class_names))
    exp = explainer.explain_instance(text, predict_fn, num_features=LIME_NUM_FEATURES, top_labels=1)
    st.components.v1.html(exp.as_html(), height=650, scrolling=True)
    label_id = exp.available_labels()[0]
    st.dataframe(pd.DataFrame(exp.as_list(label=label_id), columns=["token", "lime_weight"]), use_container_width=True)


def show_shap(text, predict_fn, class_names):
    if not SHAP_AVAILABLE:
        st.error("SHAP is not installed. Run: pip install shap")
        return
    try:
        masker = shap.maskers.Text(r"\W+")
        explainer = shap.Explainer(predict_fn, masker, output_names=list(class_names))
        shap_values = explainer([text])
        probs = predict_fn([text])[0]
        pred_idx = int(np.argmax(probs))
        st.write(f"SHAP explanation for predicted class: **{class_names[pred_idx]}**")
        try:
            html_obj = shap.plots.text(shap_values[:, :, pred_idx], display=False)
            st.components.v1.html(html_obj, height=500, scrolling=True)
        except Exception:
            tokens = shap_values.data[0]
            vals = shap_values.values[0, :, pred_idx]
            table = pd.DataFrame({"token": tokens, "shap_value": vals})
            table = table.reindex(table["shap_value"].abs().sort_values(ascending=False).index).head(25)
            st.dataframe(table, use_container_width=True)
    except Exception as e:
        st.warning("SHAP could not render for this model/text. Try TF-IDF first or install/update SHAP.")
        st.code(str(e))

# =========================
# Main application
# =========================
if "trained" not in st.session_state:
    st.session_state.trained = False
    st.session_state.models = {}
    st.session_state.results = {}
    st.session_state.data = None

with st.sidebar:
    st.header("Fixed configuration")
    st.write(f"Train: `{TRAIN_FILE}`")
    st.write(f"Evaluation: `{EVAL_FILE}`")
    st.write(f"Test: `{TEST_FILE}`")
    st.write(f"Text column: `{TEXT_COLUMN}`")
    st.write(f"Label column: `{LABEL_COLUMN}`")
    st.write(f"Min words: `{MIN_WORDS}`")
    st.write(f"Max words: `{MAX_WORDS}`")
    st.write(f"TF-IDF + SVM: `{RUN_TFIDF_SVM}`")
    st.write(f"CNN-LSTM: `{RUN_CNN_LSTM}`")
    st.write(f"mBERT: `{RUN_MBERT}`")
    train_clicked = st.button("Preprocess and train", type="primary")

if train_clicked:
    try:
        train_raw, eval_raw, test_raw = load_data_from_same_directory()

        train_df = preprocess_dataframe(train_raw, "train")
        eval_df = preprocess_dataframe(eval_raw, "evaluation")
        test_df = preprocess_dataframe(test_raw, "test")

        if BALANCE_TRAINING_CLASSES:
            train_df = balance_training_data(train_df)

        # Remove evaluation/test classes not found in training
        train_classes = set(train_df[LABEL_COLUMN].astype(str))
        eval_df = eval_df[eval_df[LABEL_COLUMN].astype(str).isin(train_classes)].copy()
        test_df = test_df[test_df[LABEL_COLUMN].astype(str).isin(train_classes)].copy()
        class_names = sorted(train_df[LABEL_COLUMN].astype(str).unique())

        st.session_state.data = {"train": train_df, "eval": eval_df, "test": test_df, "class_names": class_names}
        st.session_state.models = {}
        st.session_state.results = {}

        with st.spinner("Training selected models..."):
            if RUN_TFIDF_SVM:
                model = train_tfidf_svm(train_df[TEXT_COLUMN], train_df[LABEL_COLUMN].astype(str))
                pred = model.predict(test_df[TEXT_COLUMN])
                acc, p, r, f1, report, cm = evaluate_model(test_df[LABEL_COLUMN].astype(str), pred, class_names)
                st.session_state.models["TF-IDF + SVM"] = {"model": model}
                st.session_state.results["TF-IDF + SVM"] = {"accuracy": acc, "precision": p, "recall": r, "f1": f1, "report": report, "cm": cm, "pred": pred, "length": length_robustness(test_df, pred)}

            if RUN_CNN_LSTM:
                model, tokenizer, le, hist = train_cnn_lstm(train_df[TEXT_COLUMN], train_df[LABEL_COLUMN].astype(str), eval_df[TEXT_COLUMN], eval_df[LABEL_COLUMN].astype(str))
                probs = cnn_predict_proba(test_df[TEXT_COLUMN], model, tokenizer)
                pred = le.inverse_transform(np.argmax(probs, axis=1))
                cnn_classes = list(le.classes_)
                acc, p, r, f1, report, cm = evaluate_model(test_df[LABEL_COLUMN].astype(str), pred, cnn_classes)
                st.session_state.models["CNN-LSTM"] = {"model": model, "tokenizer": tokenizer, "label_encoder": le}
                st.session_state.results["CNN-LSTM"] = {"accuracy": acc, "precision": p, "recall": r, "f1": f1, "report": report, "cm": cm, "pred": pred, "length": length_robustness(test_df, pred), "history": hist}

            if RUN_MBERT:
                trainer, tokenizer, le = train_mbert(train_df, eval_df)
                probs = mbert_predict_proba(test_df[TEXT_COLUMN], trainer, tokenizer)
                pred = le.inverse_transform(np.argmax(probs, axis=1))
                bert_classes = list(le.classes_)
                acc, p, r, f1, report, cm = evaluate_model(test_df[LABEL_COLUMN].astype(str), pred, bert_classes)
                st.session_state.models["mBERT"] = {"trainer": trainer, "tokenizer": tokenizer, "label_encoder": le}
                st.session_state.results["mBERT"] = {"accuracy": acc, "precision": p, "recall": r, "f1": f1, "report": report, "cm": cm, "pred": pred, "length": length_robustness(test_df, pred)}

        st.session_state.trained = True
        st.success("Training complete.")

    except Exception as e:
        st.error("Training failed.")
        st.code(str(e))

if not st.session_state.trained:
    st.info("Your CSV files are read automatically from the same folder as app.py. Click 'Preprocess and train'.")
    st.markdown("""
### Expected files in this folder
- `train.csv`
- `evaluation.csv`
- `test.csv`

### Expected columns
- `text`
- `author` by default. If your label column is `author_id`, change `LABEL_COLUMN = "author_id"` at the top of `app.py`.
""")
else:
    train_df = st.session_state.data["train"]
    eval_df = st.session_state.data["eval"]
    test_df = st.session_state.data["test"]
    class_names = st.session_state.data["class_names"]

    tab_data, tab_results, tab_predict, tab_explain, tab_exports = st.tabs(["Preprocessed data", "Results", "Predict", "LIME + SHAP", "Exports"])

    with tab_data:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Train rows", len(train_df))
        c2.metric("Evaluation rows", len(eval_df))
        c3.metric("Test rows", len(test_df))
        c4.metric("Authors/classes", len(class_names))

        st.subheader("Training class distribution")
        st.dataframe(train_df[LABEL_COLUMN].value_counts().rename_axis("author").reset_index(name="count"), use_container_width=True)

        st.subheader("Length buckets")
        st.dataframe(train_df.groupby("length_bucket").agg(samples=(TEXT_COLUMN, "count"), average_words=("word_count", "mean")).reset_index(), use_container_width=True)

        st.subheader("Preprocessed preview")
        st.dataframe(train_df.head(30), use_container_width=True)

    with tab_results:
        summary = []
        for name, r in st.session_state.results.items():
            summary.append({"model": name, "accuracy": r["accuracy"], "macro_precision": r["precision"], "macro_recall": r["recall"], "macro_f1": r["f1"]})
        st.subheader("Model comparison")
        st.dataframe(pd.DataFrame(summary), use_container_width=True)

        selected = st.selectbox("Detailed model", list(st.session_state.results.keys()))
        r = st.session_state.results[selected]
        report_classes = [x for x in r["report"].keys() if x not in ["accuracy", "macro avg", "weighted avg"]]
        st.pyplot(plot_cm(r["cm"], report_classes, selected))
        st.subheader("Classification report")
        st.dataframe(pd.DataFrame(r["report"]).transpose(), use_container_width=True)
        st.subheader("Text-length robustness")
        st.dataframe(r["length"], use_container_width=True)

        if r["accuracy"] < 0.90:
            st.warning("Accuracy is below 90% on this split. Check data balance, train/test leakage, duplicate removal, and whether the labels match the intended authors.")

    def get_predict_fn_and_classes(model_name):
        info = st.session_state.models[model_name]
        if model_name == "TF-IDF + SVM":
            m = info["model"]
            return lambda texts: m.predict_proba([clean_text(t) for t in texts]), list(m.classes_)
        if model_name == "CNN-LSTM":
            return lambda texts: cnn_predict_proba([clean_text(t) for t in texts], info["model"], info["tokenizer"]), list(info["label_encoder"].classes_)
        return lambda texts: mbert_predict_proba([clean_text(t) for t in texts], info["trainer"], info["tokenizer"]), list(info["label_encoder"].classes_)

    with tab_predict:
        selected_model = st.selectbox("Model", list(st.session_state.models.keys()), key="predict_model")
        user_text = st.text_area("Enter Afrikaans text", height=180)
        if st.button("Predict author") and user_text.strip():
            predict_fn, classes = get_predict_fn_and_classes(selected_model)
            cleaned = clean_text(user_text)
            probs = predict_fn([cleaned])[0]
            pred_idx = int(np.argmax(probs))
            st.success(f"Predicted author: {classes[pred_idx]} | Confidence: {probs[pred_idx]:.4f}")
            st.dataframe(pd.DataFrame({"author": classes, "probability": probs}).sort_values("probability", ascending=False), use_container_width=True)

    with tab_explain:
        selected_model = st.selectbox("Model to explain", list(st.session_state.models.keys()), key="explain_model")
        default_text = test_df.iloc[0][TEXT_COLUMN] if len(test_df) else ""
        explain_text = st.text_area("Text to explain", value=default_text, height=180)
        method = st.radio("Explainability method", ["LIME", "SHAP"], horizontal=True)

        if st.button("Generate explanation") and explain_text.strip():
            predict_fn, classes = get_predict_fn_and_classes(selected_model)
            cleaned = clean_text(explain_text)
            probs = predict_fn([cleaned])[0]
            pred_idx = int(np.argmax(probs))
            st.info(f"Prediction explained: {classes[pred_idx]} | Confidence: {probs[pred_idx]:.4f}")
            if method == "LIME":
                show_lime(cleaned, predict_fn, classes)
            else:
                show_shap(cleaned, predict_fn, classes)

    with tab_exports:
        st.subheader("Export reports")
        if st.button("Save reports to streamlit_outputs"):
            train_df.to_csv(OUTPUT_DIR / "preprocessed_train.csv", index=False, encoding="utf-8-sig")
            eval_df.to_csv(OUTPUT_DIR / "preprocessed_evaluation.csv", index=False, encoding="utf-8-sig")
            test_df.to_csv(OUTPUT_DIR / "preprocessed_test.csv", index=False, encoding="utf-8-sig")

            rows = []
            for model_name, r in st.session_state.results.items():
                rows.append({"model": model_name, "accuracy": r["accuracy"], "macro_precision": r["precision"], "macro_recall": r["recall"], "macro_f1": r["f1"]})
                pd.DataFrame(r["report"]).transpose().to_csv(OUTPUT_DIR / f"{model_name}_classification_report.csv", encoding="utf-8-sig")
                pd.DataFrame(r["cm"]).to_csv(OUTPUT_DIR / f"{model_name}_confusion_matrix.csv", index=False, encoding="utf-8-sig")
                r["length"].to_csv(OUTPUT_DIR / f"{model_name}_length_robustness.csv", index=False, encoding="utf-8-sig")
                pred_df = test_df[[TEXT_COLUMN, LABEL_COLUMN, "word_count", "length_bucket"]].copy()
                pred_df["predicted_author"] = r["pred"]
                pred_df["correct"] = pred_df[LABEL_COLUMN].astype(str) == pred_df["predicted_author"].astype(str)
                pred_df.to_csv(OUTPUT_DIR / f"{model_name}_test_predictions.csv", index=False, encoding="utf-8-sig")

            pd.DataFrame(rows).to_csv(OUTPUT_DIR / "model_comparison_summary.csv", index=False, encoding="utf-8-sig")
            st.success(f"Saved to: {OUTPUT_DIR.resolve()}")
