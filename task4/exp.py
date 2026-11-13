"""Task 4 — meria zlepšenia klasifikátora listov jednu zmenu naraz.

Vyjadrené tým najjednoduchším spôsobom, ako v zápisníku (Task 2 a 3):
ImageFolder + DataLoader + mobilenet_v3_small (alebo LeafNet z Task 3) +
CrossEntropyLoss + Adam + slučka cez epochy. Jediná vec navyše je slovník
VARIANTS — každý riadok tabuľky je jeden taký slovník, ktorý prepíše iba to,
čo tá ktorá zmena potrebuje. Nič sa nerefaktorilovalo, žiadne triedy navyše.

Meria sa na troch množinách, všetky predpočítané raz a zdieľané všetkými
variantmi, aby sa porovnávali navzájom porovnateľne:
  val      - validačné fotky našej skupiny (62)
  camera   - tie isté fotky, raz náhodne orezané/otočené/presfarbené (pevný seed)
             -> aproximuje pohľad kamery telefónu
  hist     - fotky z minulých rokov (120), od iných ľudí a iných telefónov

Základ (baseline) je model z Task 2, teda presne ten, ktorý beží na stránke.

Použitie:
  python -I exp.py --variants baseline --seeds 0
  python -I exp.py --variants baseline,color,both --seeds 0 1 2 --out results.jsonl
"""

import argparse
import copy
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import datasets, models, transforms
from PIL import Image

# cesty a predspracovanie — rovnaké ako v zápisníku
GROUP = Path("C:/KIKI_SKOLA/MPC-MLR/ex8/task4/data/leaves")
HIST = Path("C:/KIKI_SKOLA/MPC-MLR/ex8/task4/data/hist/leaves")
SIZE = 96
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
CLASSES = ["apple", "cherry", "chestnut", "maple"]
CAMERA_SEED = 12345  # seed, pod ktorým sa raz vygeneruje "camera" množina

GEOM = [
    transforms.RandomResizedCrop(SIZE, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
]
COLOR = [transforms.ColorJitter(0.3, 0.3, 0.2, 0.05)]
TO_TENSOR = [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]


def train_transform(kind):
    """Augmentácia trénovacích dát. Validácia sa augmentuje inak ako v 3a."""
    if kind == "none":
        pre = []
    elif kind == "geom":
        pre = GEOM
    elif kind == "color":
        pre = COLOR
    elif kind == "both":
        pre = GEOM + COLOR
    else:
        raise ValueError(kind)
    return transforms.Compose(pre + TO_TENSOR)


class LeafNetBlocks(nn.Module):
    """Sieť z 3b/3d presne ako v zápisníku, len s dvoma voliteľnými vrstvami navyše."""

    def __init__(self, bn=False, gap=False):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn = bn
        if bn:
            self.bn1, self.bn2 = nn.BatchNorm2d(16), nn.BatchNorm2d(32)
            self.bn3, self.bn4 = nn.BatchNorm2d(64), nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.gap = gap
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc1 = None if gap else nn.Linear(64 * 6 * 6, 64)
        self.fc2 = nn.Linear(64, 4)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))) if self.bn else self.conv1(x))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))) if self.bn else self.conv2(x))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))) if self.bn else self.conv3(x))
        x = self.pool(torch.relu(self.bn4(self.conv4(x))) if self.bn else self.conv4(x))
        if self.gap:
            return self.fc2(self.avg(x).flatten(1))
        x = x.flatten(1)
        return self.fc2(torch.relu(self.fc1(x)))


def build_model(cfg):
    if cfg["net"] == "mobilenet":
        m = models.mobilenet_v3_small(weights="DEFAULT")
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, len(CLASSES))
        if cfg["dropout"]:
            m.classifier[2] = nn.Dropout(cfg["dropout"])
        # GlobalAvgPool tu už je súčasťou architektúry (AdaptiveAvgPool2d(1)
        # na konci features), preto variants "gap" patrí len na LeafNet.
        return m
    return LeafNetBlocks(bn=cfg["net"] == "leafnet_bn", gap=cfg["gap"])


def datasets_for(cfg, transform):
    train = datasets.ImageFolder(GROUP / "train", transform=transform)
    if cfg["extra_hist"]:
        train = ConcatDataset([train, datasets.ImageFolder(HIST / "train", transform=transform)])
    return train


@torch.no_grad()
def accuracy(model, x, y, device):
    model.eval()
    correct = 0
    for i in range(0, len(x), 64):
        pred = model(x[i : i + 64].to(device)).argmax(dim=1).cpu()
        correct += (pred == y[i : i + 64]).sum().item()
    return correct / len(x)


def preload(folder, transform, device=None):
    """Všetky obrázky množiny naraz do jedného tenzora (sú malé, vyhodnocovanie je potom okamžité)."""
    ds = datasets.ImageFolder(folder, transform=transform)
    x = torch.stack([ds[i][0] for i in range(len(ds))])
    y = torch.tensor([ds[i][1] for i in range(len(ds))])
    return x, y


def camera_like(val_folder, seed=CAMERA_SEED):
    """Validačné fotky raz upravené tak, aby pripomínali pohľad kamery: orezanie, otočenie, farby."""
    torch.manual_seed(seed)
    tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(SIZE, scale=(0.5, 1.0)),
            transforms.RandomRotation(30),
            transforms.ColorJitter(0.3, 0.3, 0.2, 0.05),
        ]
        + TO_TENSOR
    )
    return preload(val_folder, tf)


def hist_clean_subset():
    """Indexy fotiek z hist/val, ktoré nie sú kópiou fotky z hist/train.

    audit.py zistil, že v zbierke z minulých ročníkov sa 4 fotky opakujú v
    tréningu aj vo validácii (maple_161/057/047/012.jpg sú tie isté obrázky ako
    maple_028/029/004/027.jpg z val). Pre varianty, ktoré trénujú aj na hist,
    meriam hist aj bez týchto fotiek, aby nebola presnosť nadhodnotená.
    Zoznam duplikátov vytvoril audit.py do hist_dups.json (podľa obsahu, nie názvu).
    """
    dup = set(json.load(open(Path(__file__).resolve().parent / "hist_dups.json", encoding="utf-8")))
    ds = datasets.ImageFolder(HIST / "val")
    return [i for i, (p, _) in enumerate(ds.samples) if Path(p).name not in dup]


def run(name, cfg, seed, sets, device, epochs_log=False):
    torch.manual_seed(seed)
    train_ds = datasets_for(cfg, train_transform(cfg["aug"]))
    loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    model = build_model(cfg).to(device)
    if cfg["freeze"]:
        for p in model.parameters():
            p.requires_grad_(False)
        for p in model.classifier[3].parameters():
            p.requires_grad_(True)

    params = [p for p in model.parameters() if p.requires_grad]
    if cfg["optim"] == "adamw":
        optimizer = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    else:
        optimizer = torch.optim.Adam(params, lr=cfg["lr"])
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
        if cfg["scheduler"]
        else None
    )

    x_val, y_val = sets["val"]
    curve = []
    best = {"acc": -1.0, "epoch": 0, "weights": None}
    t0 = time.time()

    for epoch in range(cfg["epochs"]):
        model.train()
        loss_sum = 0.0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(images)
        if scheduler:
            scheduler.step()

        acc = accuracy(model, x_val, y_val, device)
        curve.append((epoch + 1, loss_sum / len(train_ds), acc))
        if acc > best["acc"]:  # "best epoch" — uchováme váhy najlepšej epochy
            best = {"acc": acc, "epoch": epoch + 1, "weights": copy.deepcopy(model.state_dict())}
        if epochs_log:
            print(f"  epocha {epoch + 1:2d}  loss {loss_sum / len(train_ds):.4f}  val {acc:.3f}", flush=True)

    # Hlavné čísla (val, camera, hist) VŽDY s váhami poslednej epochy — presne tak
    # vznikol aj model z Task 2, ten si počas tréningu nič nevyberal (zápisník
    # skončil na 15. epoche s val 0.855 a táto váha je nasadená).
    val_last = curve[-1][2]
    x_cam, y_cam = sets["camera"]
    x_hist, y_hist = sets["hist"]
    keep = sets["hist_clean"]  # hist bez 4 fotiek, ktoré sa opakujú v tréningu hist
    last = {
        "camera": accuracy(model, x_cam, y_cam, device),
        "hist": accuracy(model, x_hist, y_hist, device),
        "hist_excl": accuracy(model, x_hist[keep], y_hist[keep], device),
    }

    # Váhy najlepšej epochy meriam osobitne a do tej istej tabuľky ich nedávam:
    # epocha sa vyberala podľa val, takže tieto čísla sú optimistické.
    last_weights = copy.deepcopy(model.state_dict())
    model.load_state_dict(best["weights"])
    best_epoch_merenie = {
        "camera_best": accuracy(model, x_cam, y_cam, device),
        "hist_best": accuracy(model, x_hist, y_hist, device),
    }
    model.load_state_dict(last_weights)  # návrat k poslednej epoche, tou meriame

    result = {
        "variant": name,
        "seed": seed,
        "device": str(device),
        **{k: v for k, v in cfg.items()},
        "n_train": len(train_ds),
        "trainable": sum(p.numel() for p in params),
        "seconds": round(time.time() - t0, 1),
        "val_last": val_last,
        "val_best": best["acc"],
        "best_epoch": best["epoch"],
        **last,
        **best_epoch_merenie,
    }
    # model vraciam s váhami poslednej epochy, teda s tými, čo som merala
    return result, curve, model


# jeden riadok tabuľky = jeden slovník; východzie hodnoty sú baseline z Task 2
BASE = dict(
    net="mobilenet",
    aug="none",
    freeze=False,
    optim="adam",
    lr=1e-3,
    weight_decay=0.0,
    dropout=0.0,
    label_smoothing=0.0,
    scheduler=False,
    gap=False,
    extra_hist=False,
    epochs=15,
)


def variant(**changes):
    cfg = dict(BASE)
    cfg.update(changes)
    return cfg


VARIANTS = {
    # --- základ: model z Task 2, 15 epoch, bez augmentácie ---
    "baseline": variant(),
    # --- augmentácia, jedna zmena naraz ---
    "geom": variant(aug="geom"),
    "color": variant(aug="color"),
    "both_aug": variant(aug="both"),
    # --- tréningová stratégia ---
    "freeze_head": variant(freeze=True),
    "cosine30": variant(epochs=30, scheduler=True),
    "epochs30": variant(epochs=30),
    "label_smooth": variant(label_smoothing=0.1),
    "dropout40": variant(dropout=0.4),
    "adamw_wd": variant(optim="adamw", weight_decay=0.05),
    # --- dáta ---
    "hist_added": variant(extra_hist=True),
    "hist_aug": variant(extra_hist=True, aug="both"),
    # --- sieť od nuly z Task 3 ---
    "scratch": variant(net="leafnet"),
    "scratch_bn_aug": variant(net="leafnet_bn", aug="both"),
    "scratch_gap": variant(net="leafnet_bn", aug="both", gap=True),
    # --- kombinácia toho najlepšieho ---
    # combined_ac: len zmeny, ktoré samostatne namerané POMOHOLI (augmentácia
    # pomohla na kamere +0.102, cosine+30 epoch pomohol na val +0.065 aj kamere
    # +0.048). AdamW s weight decay som do nej nedal, lebo sám o sebe nič
    # neniesol (val +0.016, camera -0.005) a kombinácia má pozostávať len z
    # overených zmien. combined je pôvodná verzia so všetkými štyrmi.
    "combined_ac": variant(aug="both", epochs=30, scheduler=True),
    "combined_ac_hist": variant(aug="both", epochs=30, scheduler=True, extra_hist=True),
    "combined": variant(aug="both", epochs=30, scheduler=True, optim="adamw", weight_decay=0.05),
    "combined_hist": variant(aug="both", epochs=30, scheduler=True, optim="adamw", weight_decay=0.05, extra_hist=True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="baseline")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results.jsonl")
    ap.add_argument("--epochs-log", action="store_true")
    ap.add_argument("--epochs", type=int, default=0, help="pre rychly test prepise epochy variantu")
    ap.add_argument("--save", default="", help="subor, do ktorych ulozim vahy (len pre jeden variant a jeden seed)")
    args = ap.parse_args()

    device = torch.device(args.device)

    # vyhodnovacie množiny sa pripravia raz a sú rovnaké pre všetky varianty
    plain = transforms.Compose(TO_TENSOR)
    sets = {
        "val": preload(GROUP / "val", plain),
        "camera": camera_like(GROUP / "val"),
        "hist": preload(HIST / "val", plain),
        "hist_clean": hist_clean_subset(),
    }
    print("val", tuple(sets["val"][0].shape), "camera", tuple(sets["camera"][0].shape), "hist", tuple(sets["hist"][0].shape), flush=True)

    for name in args.variants.split(","):
        cfg = VARIANTS[name]
        if args.epochs:  # rychly test skriptu, nie vysledok pre report
            cfg = dict(cfg, epochs=args.epochs)
        for seed in args.seeds:
            result, curve, model = run(name, cfg, seed, sets, device, epochs_log=args.epochs_log)
            if args.save:  # uložím váhy poslednej epochy pre export do ONNX
                cesta = f"{args.save}_seed{seed}.pt"  # args.save je predpona, aby sa to neprepísalo
                torch.save({"state_dict": model.state_dict(), "cfg": {k: v for k, v in cfg.items()},
                            "variant": name, "seed": seed,
                            "merania": {k: result[k] for k in
                                        ("val_last", "val_best", "best_epoch", "camera", "hist",
                                         "hist_excl", "camera_best", "hist_best")}}, cesta)
                print(f"  vahy ulozene do {args.save}", flush=True)
            with open(args.out, "a", encoding="utf-8") as f:
                f.write(json.dumps(result) + "\n")
            print(
                f"{name:15s} seed{seed}  val_last {result['val_last']:.3f}  val_best {result['val_best']:.3f}"
                f" (epocha {result['best_epoch']:2d})  camera {result['camera']:.3f}  hist {result['hist']:.3f}"
                f"  {result['seconds']:.0f}s",
                flush=True,
            )
            with open(args.out + ".curve", "a", encoding="utf-8") as f:
                f.write(json.dumps({"variant": name, "seed": seed, "curve": curve}) + "\n")


if __name__ == "__main__":
    main()
