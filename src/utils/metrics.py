"""
Evaluation helpers: WER (ASR), accuracy, F1-macro (classification).
"""

from typing import Optional

import numpy as np


def compute_wer(predictions: list[str], references: list[str]) -> float:
    """
    Word Error Rate using jiwer. Lower is better.
    """
    import jiwer

    transformation = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.RemovePunctuation(),
        jiwer.ReduceToListOfListOfWords(),
    ])
    return jiwer.wer(
        references,
        predictions,
        truth_transform=transformation,
        hypothesis_transform=transformation,
    )


def compute_classification_metrics(
    y_true: list[int],
    y_pred: list[int],
    label_names: Optional[list[str]] = None,
) -> dict:
    """
    Returns accuracy, F1-macro, and per-class F1.
    """
    from sklearn.metrics import accuracy_score, classification_report, f1_score

    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    report = classification_report(
        y_true,
        y_pred,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "classification_report": report,
    }


def format_comparison_table(results: list[dict]) -> str:
    """
    Pretty-print a Markdown comparison table from a list of model result dicts.

    Expected dict keys: model_name, accuracy, f1_macro, train_time_min
    """
    header = "| Model | Accuracy | F1 (macro) | Train Time |"
    separator = "|---|---|---|---|"
    rows = [header, separator]
    for r in results:
        row = (
            f"| {r['model_name']} "
            f"| {r['accuracy']:.1%} "
            f"| {r['f1_macro']:.3f} "
            f"| ~{r.get('train_time_min', '?')} min |"
        )
        rows.append(row)
    return "\n".join(rows)
