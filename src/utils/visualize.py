"""
Visualization helpers: confusion matrix, training curves, WER comparison bar chart.
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    label_names: list[str],
    title: str = "Confusion Matrix",
    figsize: tuple = (10, 8),
) -> plt.Figure:
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    return fig


def plot_training_curves(
    train_losses: list[float],
    val_losses: list[float],
    train_accs: Optional[list[float]] = None,
    val_accs: Optional[list[float]] = None,
    title: str = "Training Curves",
) -> plt.Figure:
    has_acc = train_accs is not None and val_accs is not None
    ncols = 2 if has_acc else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 4))
    if ncols == 1:
        axes = [axes]

    epochs = range(1, len(train_losses) + 1)
    axes[0].plot(epochs, train_losses, "b-o", label="Train Loss")
    axes[0].plot(epochs, val_losses, "r-o", label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{title} — Loss")
    axes[0].legend()

    if has_acc:
        axes[1].plot(epochs, train_accs, "b-o", label="Train Acc")
        axes[1].plot(epochs, val_accs, "r-o", label="Val Acc")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title(f"{title} — Accuracy")
        axes[1].legend()

    plt.tight_layout()
    return fig


def plot_model_comparison(
    model_names: list[str],
    accuracies: list[float],
    f1_scores: list[float],
    title: str = "Model Comparison",
) -> plt.Figure:
    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, [a * 100 for a in accuracies], width, label="Accuracy (%)", color="steelblue")
    bars2 = ax.bar(x + width / 2, [f * 100 for f in f1_scores], width, label="F1 Macro (%)", color="coral")

    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_ylabel("Score (%)")
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(0, 100)

    for bar in bars1:
        ax.annotate(f"{bar.get_height():.1f}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.annotate(f"{bar.get_height():.1f}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    return fig


def plot_wer_comparison(
    model_names: list[str],
    wer_scores: list[float],
    title: str = "WER Comparison (lower is better)",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["coral" if i == 0 else "steelblue" for i in range(len(model_names))]
    bars = ax.bar(model_names, [w * 100 for w in wer_scores], color=colors)
    ax.set_ylabel("WER (%)")
    ax.set_title(title)

    for bar in bars:
        ax.annotate(f"{bar.get_height():.1f}%", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom")

    plt.tight_layout()
    return fig
