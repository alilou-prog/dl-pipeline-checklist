"""Evaluation, threshold tuning, Top-K lift, and learning-curve plotting helpers."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_predictions(name: str, y_true, y_pred, y_prob=None, split: str = "Val") -> dict:
    """Computes binary classification metrics focused on the positive class."""
    return {
        "Model": name,
        "Split": split,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision (pos)": precision_score(y_true, y_pred, zero_division=0),
        "Recall (pos)": recall_score(y_true, y_pred, zero_division=0),
        "F1 (pos)": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_prob) if y_prob is not None else np.nan,
        "PR-AUC": average_precision_score(y_true, y_prob) if y_prob is not None else np.nan,
    }


def tune_threshold(y_val, val_prob, grid=None) -> tuple[float, float]:
    """Finds the decision threshold on validation probabilities that maximizes F1."""
    thresholds = np.linspace(0.05, 0.95, 91) if grid is None else np.asarray(grid)
    scores = [f1_score(y_val, (val_prob >= t).astype(int), zero_division=0) for t in thresholds]
    best_idx = int(np.argmax(scores))
    return float(thresholds[best_idx]), float(scores[best_idx])


def compute_topk_lift(y_true, probs, ks=(0.10, 0.20, 0.30, 0.50)) -> pd.DataFrame:
    """Computes Precision@K, Recall@K (gains), and Lift over random for Top-K% cutoffs."""
    y_arr = np.asarray(y_true, dtype=int)
    base_rate = float(y_arr.mean())
    total_pos = int(y_arr.sum())
    order = np.argsort(-np.asarray(probs))
    y_sorted = y_arr[order]

    rows = []
    for k in ks:
        n_top = int(len(y_sorted) * k)
        hits = int(y_sorted[:n_top].sum())
        prec_k = hits / n_top
        rows.append({
            "Top %": f"{int(k * 100)}%",
            "Contacted": n_top,
            "Positives Caught": hits,
            "Precision@K": prec_k,
            "Recall@K (Gains)": hits / total_pos,
            "Lift": prec_k / base_rate,
        })
    return pd.DataFrame(rows).set_index("Top %")


def plot_gains_and_lift(y_true, model_probs: dict, title_suffix: str = ""):
    """Plots Cumulative Gains and Cumulative Lift curves for one or more models."""
    y_arr = np.asarray(y_true, dtype=int)
    base_rate = float(y_arr.mean())
    total_pos = int(y_arr.sum())
    pct_grid = np.linspace(0.05, 1.0, 20)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.0))
    for name, probs in model_probs.items():
        y_sorted = y_arr[np.argsort(-np.asarray(probs))]
        gains = [y_sorted[:int(len(y_sorted) * p)].sum() / total_pos for p in pct_grid]
        lifts = [y_sorted[:int(len(y_sorted) * p)].mean() / base_rate for p in pct_grid]
        axes[0].plot(pct_grid * 100, np.array(gains) * 100, marker="o", markersize=3, label=name)
        axes[1].plot(pct_grid * 100, lifts, marker="o", markersize=3, label=name)

    axes[0].plot([0, 100], [0, 100], "k--", alpha=0.5, label="Random")
    axes[0].set_title(f"Cumulative Gains Curve {title_suffix}".strip())
    axes[0].set_xlabel("% Contacted")
    axes[0].set_ylabel("% Positives Reached (Recall)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].axhline(1.0, color="black", linestyle="--", alpha=0.5, label="Random (Lift=1.0)")
    axes[1].set_title(f"Cumulative Lift Curve {title_suffix}".strip())
    axes[1].set_xlabel("% Contacted")
    axes[1].set_ylabel("Lift over Random")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    return fig, axes


def plot_learning_curves(histories: dict, title_prefix: str = ""):
    """Plots side-by-side training and validation loss and PR-AUC curves."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    colors = plt.cm.tab10.colors

    for idx, (label, hist) in enumerate(histories.items()):
        c = colors[idx % len(colors)]
        epochs = range(1, len(hist.history["loss"]) + 1)
        axes[0].plot(epochs, hist.history["loss"], color=c, linestyle="--", alpha=0.65, label=f"{label} (train)")
        axes[0].plot(epochs, hist.history["val_loss"], color=c, linewidth=2, label=f"{label} (val)")
        axes[1].plot(epochs, hist.history["pr_auc"], color=c, linestyle="--", alpha=0.65, label=f"{label} (train)")
        axes[1].plot(epochs, hist.history["val_pr_auc"], color=c, linewidth=2, label=f"{label} (val)")

    axes[0].set_title(f"{title_prefix}: Binary Cross-Entropy Loss".strip(": "))
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title(f"{title_prefix}: PR-AUC (Minority Class)".strip(": "))
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("PR-AUC")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)

    plt.tight_layout()
    return fig, axes
