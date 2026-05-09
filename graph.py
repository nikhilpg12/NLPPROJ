# %%
import json
import os

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from umap import UMAP

# Create out/ dir
os.makedirs("out", exist_ok=True)

pred_df   = pd.read_csv("out/test_predictions.csv")
latent_df = pd.read_csv("out/latent_activations.csv")
cm_matrix = np.load("out/confusion_matrix.npy")
with open("out/class_names.json") as f:
    class_names = json.load(f)
with open("out/history.json") as f:
    history = json.load(f)

short_names = [n.replace("_", " ").title() for n in class_names]

# %%  1 — Latent-space clustering coloured by author
feat_cols = [c for c in latent_df.columns if c.startswith("d")]
reducer   = UMAP(n_components=2, random_state=42)
embedding = reducer.fit_transform(latent_df[feat_cols].values)

authors  = latent_df["author"].values
unique_a = sorted(set(authors))
palette  = cm.get_cmap("tab10", len(unique_a))
color_map = {a: palette(i) for i, a in enumerate(unique_a)}

fig, ax = plt.subplots(figsize=(10, 7))
for author in unique_a:
    mask = authors == author
    ax.scatter(
        embedding[mask, 0], embedding[mask, 1],
        c=[color_map[author]], label=author.replace("_", " ").title(),
        alpha=0.6, s=18, linewidths=0,
    )
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
ax.set_title("Latent Space Clustering of Authors' Texts (UMAP)")
ax.set_xlabel("UMAP Reduced Dimension X"); ax.set_ylabel("UMAP Reduced Dimension Y")
plt.tight_layout()
plt.savefig("out/plot_latent_space.png", dpi=150)
plt.show()

# %%  2 — Per-author accuracy histogram with sample-count overlay
author_stats = (
    pred_df.groupby("true_author")
    .agg(accuracy=("correct", "mean"), n_samples=("correct", "count"))
    .reset_index()
    .sort_values("accuracy", ascending=False)
)
author_stats["short"] = author_stats["true_author"].str.replace("_", " ").str.title()

fig, ax1 = plt.subplots(figsize=(12, 5))
bars = ax1.bar(author_stats["short"], author_stats["accuracy"],
               color=sns.color_palette("viridis", len(author_stats)), edgecolor="white")
ax1.set_ylim(0, 1.12)
ax1.set_ylabel("Accuracy")
ax1.set_title("Per-Author Prediction Accuracy (bars) vs. Sample Count (line)")
ax1.tick_params(axis="x", rotation=45)

for bar, acc in zip(bars, author_stats["accuracy"]):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
             f"{acc:.0%}", ha="center", va="bottom", fontsize=8)

ax2 = ax1.twinx()
ax2.plot(author_stats["short"], author_stats["n_samples"],
         color="crimson", marker="o", linewidth=2, label="# samples")
ax2.set_ylabel("Number of Test Samples", color="crimson")
ax2.tick_params(axis="y", labelcolor="crimson")
ax2.legend(loc="upper right")

plt.tight_layout()
plt.savefig("out/plot_author_accuracy.png", dpi=150)
plt.show()

# %%  3 — Training & validation loss / accuracy curves
epochs = range(1, len(history["loss"]) + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(epochs, history["loss"],     label="Train loss",  color="steelblue")
ax1.plot(epochs, history["val_loss"], label="Val loss",    color="tomato", linestyle="--")
ax1.set_title("Loss per Epoch"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
ax1.legend()

ax2.plot(epochs, history["accuracy"],     label="Train acc",  color="steelblue")
ax2.plot(epochs, history["val_accuracy"], label="Val acc",    color="tomato", linestyle="--")
ax2.set_title("Accuracy per Epoch"); ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
ax2.legend()

plt.suptitle("CNN Training History", fontsize=13)
plt.tight_layout()
plt.savefig("out/plot_training_history.png", dpi=150)
plt.show()

# %%  4 — Confusion matrix heatmap
fig, ax = plt.subplots(figsize=(max(8, len(class_names)), max(6, len(class_names) - 1)))
sns.heatmap(
    cm_matrix, annot=True, fmt="d", cmap="Blues",
    xticklabels=short_names, yticklabels=short_names,
    linewidths=0.5, ax=ax,
)
ax.set_xlabel("Predicted Author"); ax.set_ylabel("True Author")
ax.set_title("Confusion Matrix (Test Set)")
plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("out/plot_confusion_matrix.png", dpi=150)
plt.show()

# %%  5 — Per-author prediction confidence distribution (violin plot)
pred_df["short_author"] = pred_df["true_author"].str.replace("_", " ").str.title()
order = (
    pred_df.groupby("short_author")["confidence"]
    .median().sort_values(ascending=False).index.tolist()
)

fig, ax = plt.subplots(figsize=(13, 5))
sns.violinplot(
    data=pred_df, x="short_author", y="confidence", order=order,
    hue="correct", hue_order=[1, 0],
    palette={1: "mediumseagreen", 0: "salmon"},
    split=True, inner="quart", linewidth=0.8, ax=ax,
)
ax.set_title("Prediction Confidence Distribution per Author\n(green = correct, red = incorrect)")
ax.set_xlabel("Author"); ax.set_ylabel("Softmax Confidence")
ax.tick_params(axis="x", rotation=45)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, ["Correct", "Incorrect"], title="Prediction", loc="lower right")
plt.tight_layout()
plt.savefig("out/plot_confidence_distribution.png", dpi=150)
plt.show()
