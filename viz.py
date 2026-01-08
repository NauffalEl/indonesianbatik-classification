import json, os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

WORK_DIR = Path("Code_Batik")  # samakan dengan WORK_DIR di train.py
MODELS = ["cnn", "vit", "hybrid"]  # pilih yang mau dibandingkan

def _load_json(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def main():
    rows = []
    classes_ref = None
    for m in MODELS:
        fp = WORK_DIR / "results" / m / "metrics_test.json"
        if not fp.exists():
            print(f"[SKIP] {m} tidak ditemukan: {fp}")
            continue
        data = _load_json(fp)
        acc = float(data["test"]["accuracy"])
        f1m = float(data["test"]["f1_macro"])
        classes = data.get("classes", [])
        # Ambil f1 per class dari classification_report
        report = data["test"]["classification_report"]
        f1_per_class = [report[c]["f1-score"] for c in classes]

        if classes_ref is None:
            classes_ref = classes
        elif classes != classes_ref:
            print("[WARN] Urutan/isi kelas beda antar model. Hasil per-class mungkin tak sebanding.")
        rows.append({
            "model": m,
            "accuracy": acc,
            "f1_macro": f1m,
            "classes": classes,
            "f1_per_class": f1_per_class,
        })

    if not rows:
        print("Tidak ada hasil yang bisa dibandingkan.")
        return

    out_dir = WORK_DIR / "comparison"
    _ensure_dir(out_dir)

    # 1) Bar Accuracy
    names = [r["model"] for r in rows]
    accs  = [r["accuracy"] for r in rows]
    fig1 = plt.figure(figsize=(8,5))
    ax1 = plt.gca()
    ax1.bar(np.arange(len(names)), accs)
    ax1.set_xticks(np.arange(len(names))); ax1.set_xticklabels(names)
    ax1.set_ylim(0,1.0)
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Accuracy Comparison (Test)")
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    fig1.tight_layout()
    p1 = out_dir / "accuracy_comparison.png"
    fig1.savefig(p1, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig1)
    print("[SAVE]", p1)

    # 2) Bar F1-macro
    f1m  = [r["f1_macro"] for r in rows]
    fig2 = plt.figure(figsize=(8,5))
    ax2 = plt.gca()
    ax2.bar(np.arange(len(names)), f1m)
    ax2.set_xticks(np.arange(len(names))); ax2.set_xticklabels(names)
    ax2.set_ylim(0,1.0)
    ax2.set_ylabel("F1-macro")
    ax2.set_title("F1-macro Comparison (Test)")
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    fig2.tight_layout()
    p2 = out_dir / "f1macro_comparison.png"
    fig2.savefig(p2, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig2)
    print("[SAVE]", p2)

    # 3) Grouped bar F1 per class
    if classes_ref:
        n_cls = len(classes_ref)
        x = np.arange(n_cls)
        width = min(0.8 / len(rows), 0.25)  # lebar bar per model
        fig3 = plt.figure(figsize=(max(12, 0.6*n_cls), 6))
        ax3 = plt.gca()
        for i, r in enumerate(rows):
            offs = (i - (len(rows)-1)/2)*width
            ax3.bar(x + offs, r["f1_per_class"], width=width, label=r["model"])
        ax3.set_xticks(x); ax3.set_xticklabels(classes_ref, rotation=45, ha="right")
        ax3.set_ylim(0, 1.0)
        ax3.set_ylabel("F1-score")
        ax3.set_title("Per-class F1 Comparison")
        ax3.grid(axis="y", linestyle="--", alpha=0.4)
        ax3.legend()
        fig3.tight_layout()
        p3 = out_dir / "per_class_f1_comparison.png"
        fig3.savefig(p3, dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(fig3)
        print("[SAVE]", p3)

    # 4) Simpan ringkas CSV
    csv_path = out_dir / "summary.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("model,accuracy,f1_macro\n")
        for r in rows:
            f.write(f'{r["model"]},{r["accuracy"]:.6f},{r["f1_macro"]:.6f}\n')
    print("[SAVE]", csv_path)

if __name__ == "__main__":
    main()
