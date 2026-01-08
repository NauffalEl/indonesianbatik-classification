import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def load_metrics(json_path):
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    classes = data["classes"]
    cm_raw  = np.array(data["test"]["confusion_matrix"], dtype=float)
    report  = data["test"]["classification_report"]
    f1_map = {c: float(report.get(c, {}).get("f1-score", 0.0)) for c in classes}
    return classes, cm_raw, f1_map

def pick_topk(classes, cm_raw, f1_map, topk=15, mode="error"):
    row_sum = cm_raw.sum(axis=1, keepdims=True)
    cm = np.divide(cm_raw, row_sum, out=np.zeros_like(cm_raw), where=row_sum!=0)
    diag = np.diag(cm)
    err  = 1.0 - diag
    scores = np.array([1.0 - f1_map[c] for c in classes]) if mode=="lowf1" else err
    idx = np.argsort(scores)[::-1][:topk]
    sel = [classes[i] for i in idx]
    return idx, sel, cm

def plot_confusion_subset(classes, cm, idx, sel, out_path, figsize=(8,8), dpi=300):
    cm_sub = cm[np.ix_(idx, idx)]
    plt.rcParams.update({"font.size":8})
    fig = plt.figure(figsize=figsize, dpi=dpi, constrained_layout=True)
    ax = sns.heatmap(
        cm_sub, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=sel, yticklabels=sel, cbar=True,
        annot_kws={"size":7, "color":"white"}
    )
    ax.set_title("Normalized Confusion Matrix (Top-15 hardest classes)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=str, default=str(Path(__file__).resolve().parents[1] / "Code_Batik" / "results" / "hybrid" / "metrics_test.json"))
    ap.add_argument("--out",  type=str, default=str(Path(__file__).resolve().parents[1] / "Code_Batik" / "results" / "hybrid" / "confmat_top15.png"))
    ap.add_argument("--topk", type=int, default=15)
    ap.add_argument("--mode", type=str, choices=["error","lowf1"], default="error")
    args = ap.parse_args()

    classes, cm_raw, f1_map = load_metrics(Path(args.json))
    idx, sel, cm = pick_topk(classes, cm_raw, f1_map, args.topk, args.mode)
    plot_confusion_subset(classes, cm, idx, sel, Path(args.out))

if __name__ == "__main__":
    main()
