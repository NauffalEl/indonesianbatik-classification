import csv
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

from .model.cnn_model import EffB2_CBAM_Arc

PKG_DIR = Path(__file__).resolve().parent          # .../BATIK/CNN_Batik
ROOT    = PKG_DIR.parent                           # .../BATIK
DATA_DIR = ROOT / "data_testing"                   # <- sumber gambar
ALLOWED = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

IMG_SIZE = 224
TOPK = 3

def _find_images(folder: Path) -> List[Path]:
    return [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in ALLOWED]

def _prep(img_size: int):
    from torchvision import transforms
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

def _guess_model_path() -> Path:
    cands = [
        PKG_DIR / "checkpoints" / "best.pt",
        PKG_DIR / "checkpoints" / "last.pt",
    ]
    for c in cands:
        if c.exists():
            return c
    return cands[0]

def _load_ckpt(model_path: Path, device):
    # set explicit weights_only=False untuk hilangkan FutureWarning default
    ckpt = torch.load(str(model_path), map_location=device, weights_only=False)
    cls = ckpt.get("class_names", None)
    state = ckpt["model"] if "model" in ckpt else ckpt
    return state, cls

def _read_tensor(p: Path, tfm):
    img = Image.open(p).convert("RGB")
    return tfm(img).unsqueeze(0)

@torch.no_grad()
def predict_paths(model, paths: List[Path], img_size: int, device, class_names, topk: int
                  ) -> List[Tuple[str, List[Tuple[str, float]]]]:
    tfm = _prep(img_size)
    model.eval()
    results = []
    for p in paths:
        try:
            xb = _read_tensor(p, tfm).to(device)
        except Exception as e:
            print(f"[SKIP] {p} ({e})")
            continue
        emb = model(xb, labels=None)
        W = model.arc.weight
        logits = F.linear(F.normalize(emb), F.normalize(W))
        probs = F.softmax(logits, dim=1).squeeze(0).cpu()
        k = min(topk, probs.numel())
        topv, topi = torch.topk(probs, k=k)
        preds = [(class_names[i], float(v)) for v, i in zip(topv, topi)]
        results.append((str(p), preds))
    return results

def save_csv(results: List[Tuple[str, List[Tuple[str, float]]]], out_csv: Path, topk: int):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    header = ["path"] + sum(([f"top{k}_label", f"top{k}_prob"] for k in range(1, topk+1)), [])
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for p, preds in results:
            row = [p]
            for k in range(topk):
                if k < len(preds):
                    row += [preds[k][0], f"{preds[k][1]:.6f}"]
                else:
                    row += ["", ""]
            w.writerow(row)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) siapkan model
    model_path = _guess_model_path()
    if not model_path.exists():
        print(f"[ERROR] Model tidak ditemukan: {model_path}")
        print("Taruh checkpoint di CNN_Batik/checkpoints/best.pt atau last.pt")
        return
    state, classes = _load_ckpt(model_path, device)
    if not classes:
        raise RuntimeError("class_names tidak ada di checkpoint.")
    model = EffB2_CBAM_Arc(num_classes=len(classes)).to(device)
    model.load_state_dict(state)

    # 2) ambil gambar dari BATIK/data_testing
    if not DATA_DIR.exists():
        print(f"[ERROR] Folder tidak ada: {DATA_DIR}")
        return
    paths = _find_images(DATA_DIR)
    if not paths:
        print(f"[ERROR] Tidak ada gambar di {DATA_DIR}")
        return
    print(f"[INFO] Ditemukan {len(paths)} gambar di {DATA_DIR}")

    # 3) prediksi
    results = predict_paths(model, paths, IMG_SIZE, device, classes, TOPK)

    # 4) tampilkan ringkas & simpan CSV
    for p, preds in results:
        top1 = preds[0] if preds else ("", 0.0)
        pretty = ", ".join([f"{n}:{prob:.3f}" for n, prob in preds])
        print(f"{p} -> TOP1: {top1[0]} ({top1[1]:.3f}) | top{len(preds)}: {pretty}")

    out_dir = PKG_DIR / "resultpred"
    out_csv = out_dir / "predictions_data_testing.csv"
    save_csv(results, out_csv, TOPK)
    print(f"[SAVE] {out_csv}")

if __name__ == "__main__":
    main()
