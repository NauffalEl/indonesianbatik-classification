import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional
from sklearn.metrics import confusion_matrix, classification_report

def _ensure_dir(p: str):
    os.makedirs(os.path.dirname(p), exist_ok=True)

def plot_confusion_matrix(y_true, y_pred, class_names: List[str],
                          normalize: bool = True,
                          title: str = "Confusion Matrix",
                          out_path: Optional[str] = None):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype(np.float64)
        row_sum = cm.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        cm = cm / row_sum

    fig = plt.figure(figsize=(8, 7))
    ax = plt.gca()
    im = ax.imshow(cm, interpolation='nearest')
    ax.set_title(title)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    # annotate
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    if out_path:
        _ensure_dir(out_path)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    return fig

def plot_f1_bars(y_true, y_pred, class_names: List[str],
                 title: str = "F1-score per Class",
                 out_path: Optional[str] = None):
    # ambil f1 tiap kelas dari classification_report (dict)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    f1s = [report[c]["f1-score"] for c in class_names]
    macro_f1 = report["macro avg"]["f1-score"]

    fig = plt.figure(figsize=(10, 5))
    ax = plt.gca()
    ax.bar(np.arange(len(class_names)), f1s)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("F1-score")
    ax.set_title(f"{title} (macro={macro_f1:.3f})")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()

    if out_path:
        _ensure_dir(out_path)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    return fig, macro_f1
