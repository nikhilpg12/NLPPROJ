"""
Local mBERT Authorship Attribution for Afrikaans

Uses ONLY these CSV columns:
    text   -> input feature
    author -> label/class to predict

Required files in the same folder:
    train.csv
    evaluation.csv
    test.csv

First-time model download:
    python main.py --download-model

Normal local training:
    python main.py
"""

import os
import json
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed
)


# =========================
# CONFIG
# =========================

TRAIN_FILE = "train.csv"
EVAL_FILE = "evaluation.csv"
TEST_FILE = "test.csv"

HF_MODEL_ID = "bert-base-multilingual-cased"
LOCAL_MODEL_DIR = "local_mbert"

OUTPUT_DIR = "mbert_afrikaans_authorship_model"

MAX_LENGTH = 512
NUM_EPOCHS = 8
BATCH_SIZE = 4
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
SEED = 42


# =========================
# DOWNLOAD mBERT ONCE
# =========================

def download_mbert_once():
    """
    Downloads mBERT into a local folder.
    Run this once while connected to the internet.
    """

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is not installed. Run:\n"
            "pip install huggingface_hub"
        )

    print(f"\nDownloading {HF_MODEL_ID} into local folder: {LOCAL_MODEL_DIR}")

    snapshot_download(
        repo_id=HF_MODEL_ID,
        local_dir=LOCAL_MODEL_DIR
    )

    print("\nDownload complete.")
    print(f"mBERT is now saved locally in: {LOCAL_MODEL_DIR}")


# =========================
# REPRODUCIBILITY
# =========================

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================
# DATA LOADING
# =========================

def read_csv_safely(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find {path}. "
            "Make sure the CSV file is in the same folder as main.py."
        )

    print(f"Loading: {path}")

    df = pd.read_csv(path)

    # Standardise column names
    df.columns = [str(col).strip().lower() for col in df.columns]

    required_columns = ["text", "author"]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(
                f"{path} is missing the required column: {col}. "
                "The model only uses text and author."
            )

    # Use ONLY required columns
    df = df[["text", "author"]].copy()

    # Remove missing values
    df = df.dropna(subset=["text", "author"])

    # Basic cleaning
    df["text"] = df["text"].astype(str).str.strip()
    df["author"] = df["author"].astype(str).str.strip()

    # Remove empty rows
    df = df[(df["text"] != "") & (df["author"] != "")]

    # Remove accidental string nan values
    df = df[
        (df["text"].str.lower() != "nan") &
        (df["author"].str.lower() != "nan")
    ]

    return df


def load_data():
    train_df = read_csv_safely(TRAIN_FILE)
    eval_df = read_csv_safely(EVAL_FILE)
    test_df = read_csv_safely(TEST_FILE)

    print("\nDataset sizes:")
    print(f"Train:      {len(train_df)}")
    print(f"Evaluation: {len(eval_df)}")
    print(f"Test:       {len(test_df)}")

    print("\nTraining author distribution:")
    print(train_df["author"].value_counts())

    return train_df, eval_df, test_df


# =========================
# LABEL ENCODING
# =========================

def encode_labels(train_df, eval_df, test_df):
    label_encoder = LabelEncoder()

    train_df["labels"] = label_encoder.fit_transform(train_df["author"])

    known_authors = set(label_encoder.classes_)

    for split_name, df in [("evaluation", eval_df), ("test", test_df)]:
        unseen_authors = sorted(set(df["author"]) - known_authors)

        if unseen_authors:
            raise ValueError(
                f"The {split_name} set contains authors not found in training data: "
                f"{unseen_authors}"
            )

    eval_df["labels"] = label_encoder.transform(eval_df["author"])
    test_df["labels"] = label_encoder.transform(test_df["author"])

    id2label = {}
    label2id = {}

    for i, author in enumerate(label_encoder.classes_):
        id2label[i] = author
        label2id[author] = i

    print("\nLabel mapping:")
    for i, author in id2label.items():
        print(f"{i}: {author}")

    return train_df, eval_df, test_df, label_encoder, id2label, label2id


# =========================
# TOKENIZATION
# =========================

def prepare_datasets(train_df, eval_df, test_df, tokenizer):
    train_dataset = Dataset.from_pandas(
        train_df[["text", "labels"]],
        preserve_index=False
    )

    eval_dataset = Dataset.from_pandas(
        eval_df[["text", "labels"]],
        preserve_index=False
    )

    test_dataset = Dataset.from_pandas(
        test_df[["text", "labels"]],
        preserve_index=False
    )

    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH
        )

    train_dataset = train_dataset.map(tokenize_batch, batched=True)
    eval_dataset = eval_dataset.map(tokenize_batch, batched=True)
    test_dataset = test_dataset.map(tokenize_batch, batched=True)

    train_dataset = train_dataset.remove_columns(["text"])
    eval_dataset = eval_dataset.remove_columns(["text"])
    test_dataset = test_dataset.remove_columns(["text"])

    return train_dataset, eval_dataset, test_dataset


# =========================
# METRICS
# =========================

def compute_metrics(eval_pred):
    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predictions)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="macro",
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1
    }


# =========================
# TRAINING ARGUMENTS
# =========================

def create_training_arguments():
    """
    Handles different Transformers versions.
    Some versions use eval_strategy.
    Some versions use evaluation_strategy.
    """

    gpu_available = torch.cuda.is_available()

    common_args = dict(
        output_dir=OUTPUT_DIR,
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        seed=SEED,
        fp16=gpu_available,
        dataloader_pin_memory=gpu_available
    )

    try:
        return TrainingArguments(
            eval_strategy="epoch",
            **common_args
        )

    except TypeError:
        return TrainingArguments(
            evaluation_strategy="epoch",
            **common_args
        )


# =========================
# LOCAL MODEL LOADING
# =========================

def load_local_mbert(num_labels, id2label, label2id):
    if not os.path.exists(LOCAL_MODEL_DIR):
        raise FileNotFoundError(
            f"\nThe local mBERT folder '{LOCAL_MODEL_DIR}' was not found.\n\n"
            "First run this command while connected to the internet:\n"
            "python main.py --download-model\n"
        )

    print(f"\nLoading mBERT locally from: {LOCAL_MODEL_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(
        LOCAL_MODEL_DIR,
        local_files_only=True
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        LOCAL_MODEL_DIR,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        local_files_only=True,
        ignore_mismatched_sizes=True
    )

    return tokenizer, model


# =========================
# PREDICTION FUNCTION
# =========================

def predict_author(text, model, tokenizer):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    model.eval()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=-1)

    predicted_id = torch.argmax(probabilities, dim=-1).item()
    confidence = probabilities[0][predicted_id].item()

    predicted_author = model.config.id2label[predicted_id]

    return predicted_author, confidence


# =========================
# MAIN TRAINING
# =========================

def train_model():
    set_all_seeds(SEED)

    train_df, eval_df, test_df = load_data()

    train_df, eval_df, test_df, label_encoder, id2label, label2id = encode_labels(
        train_df,
        eval_df,
        test_df
    )

    num_labels = len(label_encoder.classes_)

    print(f"\nNumber of authors/classes: {num_labels}")

    tokenizer, model = load_local_mbert(
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )

    train_dataset, eval_dataset, test_dataset = prepare_datasets(
        train_df,
        eval_df,
        test_df,
        tokenizer
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    if torch.cuda.is_available():
        print("\nUsing device: GPU/CUDA")
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("\nUsing device: CPU")
        print("Note: You are training on CPU. mBERT will work, but it may be slow.")

    training_args = create_training_arguments()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=2)
        ]
    )

    print("\nStarting training...")
    trainer.train()

    print("\nEvaluating on evaluation set...")
    eval_results = trainer.evaluate(
        eval_dataset=eval_dataset,
        metric_key_prefix="eval"
    )
    print(eval_results)

    print("\nEvaluating on test set...")
    test_results = trainer.evaluate(
        eval_dataset=test_dataset,
        metric_key_prefix="test"
    )
    print(test_results)

    print("\nCreating detailed test report...")

    predictions = trainer.predict(test_dataset)

    y_pred = np.argmax(predictions.predictions, axis=-1)
    y_true = predictions.label_ids

    report = classification_report(
        y_true,
        y_pred,
        target_names=list(label_encoder.classes_),
        zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred)

    print("\nClassification report:")
    print(report)

    print("\nConfusion matrix:")
    print(cm)

    final_model_path = Path(OUTPUT_DIR) / "final_model"
    final_model_path.mkdir(parents=True, exist_ok=True)

    trainer.save_model(str(final_model_path))
    tokenizer.save_pretrained(str(final_model_path))

    with open(final_model_path / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "id2label": id2label,
                "label2id": label2id
            },
            f,
            ensure_ascii=False,
            indent=4
        )

    with open(final_model_path / "test_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    pd.DataFrame(
        cm,
        index=label_encoder.classes_,
        columns=label_encoder.classes_
    ).to_csv(
        final_model_path / "test_confusion_matrix.csv",
        encoding="utf-8-sig"
    )

    with open(final_model_path / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=4)

    # Save every test prediction
    test_predictions_df = test_df[["text", "author"]].copy()

    test_predictions_df["predicted_author"] = [
        id2label[int(prediction)]
        for prediction in y_pred
    ]

    test_predictions_df["correct"] = (
        test_predictions_df["author"] == test_predictions_df["predicted_author"]
    )

    test_predictions_df.to_csv(
        final_model_path / "test_predictions.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # Save only wrong predictions
    test_errors_df = test_predictions_df[
        test_predictions_df["correct"] == False
    ].copy()

    test_errors_df.to_csv(
        final_model_path / "test_errors.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\nDone. Final trained model saved to: {final_model_path}")

    print("\nSaved output files:")
    print(f"- {final_model_path / 'test_metrics.json'}")
    print(f"- {final_model_path / 'test_classification_report.txt'}")
    print(f"- {final_model_path / 'test_confusion_matrix.csv'}")
    print(f"- {final_model_path / 'test_predictions.csv'}")
    print(f"- {final_model_path / 'test_errors.csv'}")

    sample_text = input(
        "\nPaste Afrikaans text to predict the author, or press Enter to skip:\n\n"
    )

    if sample_text.strip() != "":
        predicted_author, confidence = predict_author(sample_text, model, tokenizer)

        print("\nPredicted author:", predicted_author)
        print("Confidence:", round(confidence, 4))


# =========================
# SCRIPT ENTRY POINT
# =========================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--download-model",
        action="store_true",
        help="Download mBERT into the local_mbert folder."
    )

    args = parser.parse_args()

    if args.download_model:
        download_mbert_once()
        return

    train_model()


if __name__ == "__main__":
    main()