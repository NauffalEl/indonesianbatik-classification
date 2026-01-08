from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageFile; ImageFile.LOAD_TRUNCATED_IMAGES = True

class ListDataset(Dataset):
    def __init__(self, items, tfm): self.items, self.tfm = items, tfm
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        p, y = self.items[idx]
        try: img = Image.open(p).convert("RGB")
        except Exception: img = Image.new("RGB", (224,224), (0,0,0))
        return (self.tfm(img) if self.tfm else img), y

def _valid_items(imgfolder):
    items = []
    for p, y in imgfolder.samples:
        items.append((p, y)) 
    return items

def build_loaders_tt(data_dir: str, img_size: int, batch_size: int, num_workers: int = 4):
    root = Path(data_dir)
    train_dir, test_dir = root/"train", root/"test"
    assert train_dir.exists() and test_dir.exists(), "Harus ada dataset_batik/train dan dataset_batik/test"

    mean, std = (0.485,0.456,0.406), (0.229,0.224,0.225)
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    tmp_train = datasets.ImageFolder(str(train_dir))
    classes   = [c for c,_ in sorted(tmp_train.class_to_idx.items(), key=lambda kv: kv[1])]
    train_ds  = ListDataset(_valid_items(tmp_train), tfm)

    tmp_test  = datasets.ImageFolder(str(test_dir))
    classes_t = [c for c,_ in sorted(tmp_test.class_to_idx.items(), key=lambda kv: kv[1])]
    assert classes == classes_t, "Urutan kelas train dan test tidak sama!"

    test_ds   = ListDataset(_valid_items(tmp_test), tfm)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader, classes
