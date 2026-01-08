# CNN_Batik/train.py
import os, csv, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from .utils.seed import set_seed
from .data.dataload import build_loaders_tt
from .model.cnn import CNNModel
from .model.vit import ViTModel
from .model.hybrid import HybridModel

#config
BASE   = Path(__file__).resolve().parents[1]
MODEL  = "cnn"  # "cnn" | "vit" | "hybrid"

PRESET = {
    "cnn":    {"IMG_SIZE": 224, "BATCH_SIZE": 32, "EPOCHS": 100, "LR": 1e-4, "NUM_WORKERS": 8},
    "vit":    {"IMG_SIZE": 224, "BATCH_SIZE": 32, "EPOCHS": 100, "LR": 1e-4, "NUM_WORKERS": 8},
    "hybrid": {"IMG_SIZE": 224, "BATCH_SIZE": 32, "EPOCHS": 100, "LR": 1e-4, "NUM_WORKERS": 8},
}
CFG = {
    "MODEL": MODEL,
    "DATA_DIR": str(BASE / "dataset_batik"),  
    "WORK_DIR": str(BASE / "Code_Batik"),
    "SEED": 42,
    "EMBED_DIM": 512,
    "ARC_S": 30.0,
    "ARC_M": 0.50,
    **PRESET[MODEL],
}

def _ensure_dir(p): os.makedirs(p, exist_ok=True)

def _build_model(name, ncls, cfg):
    if name == "cnn":
        return CNNModel(num_classes=ncls, embed_dim=cfg["EMBED_DIM"], arc_s=cfg["ARC_S"], arc_m=cfg["ARC_M"])
    if name == "vit":
        return ViTModel(num_classes=ncls, embed_dim=cfg["EMBED_DIM"], arc_s=cfg["ARC_S"], arc_m=cfg["ARC_M"], backbone="vit_b_16")
    if name == "hybrid":
        return HybridModel(num_classes=ncls, embed_dim=cfg["EMBED_DIM"], arc_s=cfg["ARC_S"], arc_m=cfg["ARC_M"],
                           cnn_backbone="resnet50", vit_backbone="vit_b_16")
    raise ValueError("MODEL harus 'cnn' | 'vit' | 'hybrid'")

def _save_confmat(y_true, y_pred, classes, out_path):
    cm = confusion_matrix(y_true, y_pred).astype(float)
    cm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix (Normalized) - Test")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(classes))); ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right"); ax.set_yticklabels(classes)
    fmt = ".2f"; th = cm.max()/2.0 if cm.size>0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > th else "black", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)

def _save_f1bar(y_true, y_pred, classes, out_path):
    rep = classification_report(y_true, y_pred, target_names=classes, output_dict=True, zero_division=0)
    f1s = [rep[c]["f1-score"] for c in classes]; m = rep["macro avg"]["f1-score"]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(np.arange(len(classes)), f1s, color="skyblue")
    ax.set_xticks(np.arange(len(classes))); ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_ylim(0, 1.0); ax.set_ylabel("F1-score"); ax.set_title(f"F1 per Class (macro={m:.3f})")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); yps, yts = [], []
    for xb, yb in tqdm(loader, desc="eval", leave=False):
        xb, yb = xb.to(device), yb.to(device)
        emb = model(xb, labels=None)
        W = model.arc.weight
        logits = F.linear(F.normalize(emb), F.normalize(W))
        yps.append(logits.argmax(1).cpu()); yts.append(yb.cpu())
    y_pred = torch.cat(yps).numpy(); y_true = torch.cat(yts).numpy()
    return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average="macro"), y_true, y_pred

def train_one_epoch(model, loader, crit, opt, device):
    model.train(); total = 0.0
    for xb, yb in tqdm(loader, desc="train", leave=False):
        xb, yb = xb.to(device), yb.to(device)
        logits, _ = model(xb, labels=yb)
        loss = crit(logits, yb)
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item() * xb.size(0)
    return total / len(loader.dataset)

if __name__ == "__main__":
    set_seed(CFG["SEED"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    train_loader, test_loader, classes = build_loaders_tt(
        data_dir=CFG["DATA_DIR"], img_size=CFG["IMG_SIZE"],
        batch_size=CFG["BATCH_SIZE"], num_workers=CFG["NUM_WORKERS"]
    )

    model = _build_model(CFG["MODEL"], ncls=len(classes), cfg=CFG).to(device)
    crit  = nn.CrossEntropyLoss()
    opt   = optim.AdamW(model.parameters(), lr=CFG["LR"])
    sch   = CosineAnnealingLR(opt, T_max=CFG["EPOCHS"])

    ckpt_dir = Path(CFG["WORK_DIR"]) / "checkpoints" / CFG["MODEL"]
    res_dir  = Path(CFG["WORK_DIR"]) / "results"     / CFG["MODEL"]
    _ensure_dir(ckpt_dir); _ensure_dir(res_dir)
    ckpt_last = ckpt_dir / "last.pt"

    for ep in range(1, CFG["EPOCHS"] + 1):
        tr_loss = train_one_epoch(model, train_loader, crit, opt, device)
        sch.step()
        print(f"[{ep:03d}] loss={tr_loss:.4f}")
        torch.save({"model": model.state_dict(), "epoch": ep, "classes": classes}, ckpt_last)

    print(f"Saved last: {ckpt_last}")
    state = torch.load(ckpt_last, map_location="cpu")
    model.load_state_dict(state["model"])

    acc, f1m, y_true, y_pred = evaluate(model, test_loader, device)
    print(f"[TEST] acc={acc:.4f} f1_macro={f1m:.4f}")

    rep = classification_report(y_true, y_pred, target_names=classes, output_dict=True, zero_division=0)
    cm  = confusion_matrix(y_true, y_pred)

    with open(res_dir / "metrics_test.json", "w", encoding="utf-8") as f:
        json.dump({"classes": classes, "test": {
            "accuracy": float(acc), "f1_macro": float(f1m),
            "classification_report": rep, "confusion_matrix": cm.astype(int).tolist()
        }}, f, ensure_ascii=False, indent=2)

    with open(res_dir / "metrics_test.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", "precision", "recall", "f1", "support"])
        for c in classes:
            m = rep.get(c, {})
            w.writerow([c, f"{m.get('precision',0):.6f}", f"{m.get('recall',0):.6f}",
                        f"{m.get('f1-score',0):.6f}", int(m.get('support',0))])
        w.writerow([])
        w.writerow(["macro avg",
                    f"{rep['macro avg']['precision']:.6f}",
                    f"{rep['macro avg']['recall']:.6f}",
                    f"{rep['macro avg']['f1-score']:.6f}",
                    int(rep['macro avg']['support'])])
        w.writerow(["weighted avg",
                    f"{rep['weighted avg']['precision']:.6f}",
                    f"{rep['weighted avg']['recall']:.6f}",
                    f"{rep['weighted avg']['f1-score']:.6f}",
                    int(rep['weighted avg']['support'])])
        w.writerow(["accuracy", f"{float(rep.get('accuracy',0)):.6f}", "", "", ""])

    _save_confmat(y_true, y_pred, classes, str(res_dir / "confusion_matrix_test.png"))
    _save_f1bar(y_true, y_pred, classes, str(res_dir / "f1_per_class_test.png"))
