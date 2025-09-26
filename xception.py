# train.py
import os, argparse, time, math, random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

# ------------------------- Utils -------------------------
def set_seed(seed: int = 42):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True  # tăng tốc cho conv

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def infer_dataset_name(root: str):
    return Path(root).resolve().name

# ------------------------- Transforms -------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def build_transforms(img_size: int):
    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tfms = transforms.Compose([
        transforms.Resize(int(img_size*1.14)),   # ≈ 256 cho 224
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tfms, eval_tfms

# ------------------------- Data -------------------------
def make_loaders(data_root: str, img_size: int, batch_size: int, num_workers: int,
                 balance: bool = False):
    train_tfms, eval_tfms = build_transforms(img_size)

    train_dir = os.path.join(data_root, "train")
    val_dir   = os.path.join(data_root, "val")
    test_dir  = os.path.join(data_root, "test")

    train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)
    val_ds   = datasets.ImageFolder(val_dir,   transform=eval_tfms)
    test_ds  = datasets.ImageFolder(test_dir,  transform=eval_tfms) if os.path.isdir(test_dir) else None

    print("Classes / idx:", train_ds.class_to_idx)

    if balance:
        # Weighted sampler nếu lệch lớp
        targets = [y for _, y in train_ds.samples]
        class_count = torch.bincount(torch.tensor(targets))
        class_weights = 1.0 / class_count.float()
        sample_weights = [class_weights[t] for t in targets]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                                  num_workers=num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True)

    val_loader  = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    test_loader = (DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
                   if test_ds is not None else None)
    num_classes = len(train_ds.classes)
    return train_loader, val_loader, test_loader, num_classes

# ------------------------- Models -------------------------
def build_model(model_name: str, num_classes: int):
    name = model_name.lower()
    if name in ["efficientnet_b3", "b3", "efficientnet"]:
        from efficientnet import EfficientNetB3 as EffB3
        model = EffB3(num_classes=num_classes)
    elif name in ["xception", "xcep"]:
        from xception import Xception
        model = Xception(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model

# ------------------------- Train / Eval -------------------------
@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += loss.item() * x.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return loss_sum/total, correct/total

def train_one_epoch(model, loader, device, optimizer, criterion, scaler=None, max_norm=None):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if scaler is None:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            if max_norm:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
        else:
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            if max_norm:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()

        loss_sum += loss.item() * x.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += x.size(0)

    return loss_sum/total, correct/total

# ------------------------- Main -------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, required=True, help="Path tới thư mục Dataset chứa train/val/(test)")
    p.add_argument("--dataset-name", type=str, default=None, help="Tên dataset để gắn vào file .pth; mặc định = tên thư mục root")
    p.add_argument("--model", type=str, required=True, choices=["efficientnet_b3","b3","xception"],
                   help="Chọn model: efficientnet_b3 hoặc xception")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=10, help="Early stopping patience (0 để tắt)")
    p.add_argument("--balance", action="store_true", help="Dùng WeightedRandomSampler nếu lệch lớp")
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--amp", action="store_true", help="Bật mixed precision")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-dir", type=str, default="checkpoints")
    args = p.parse_args()

    set_seed(args.seed)
    device = get_device()
    os.makedirs(args.save_dir, exist_ok=True)

    dataset_name = args.dataset_name or infer_dataset_name(args.data_root)
    save_path = os.path.join(args.save_dir, f"{args.model}_{dataset_name}_best.pth")

    # Data
    train_loader, val_loader, test_loader, num_classes = make_loaders(
        args.data_root, args.img_size, args.batch_size, args.workers, balance=args.balance
    )

    # Model
    model = build_model(args.model, num_classes=num_classes).to(device)

    # Loss / Optim / Sched
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler() if (args.amp and device == "cuda") else None

    # Train loop
    best_val_acc, best_state = 0.0, None
    bad_epochs = 0
    print(f"Start training on {device}. Saving best to: {save_path}")
    for epoch in range(1, args.epochs+1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, device, optimizer, criterion, scaler)
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save(best_state, save_path)
            bad_epochs = 0
            flag = " (saved)"
        else:
            bad_epochs += 1
            flag = ""
        dt = time.time() - t0
        print(f"Epoch {epoch:02d}/{args.epochs} | "
              f"train {tr_loss:.4f}/{tr_acc:.4f} | "
              f"val {val_loss:.4f}/{val_acc:.4f} | "
              f"time {dt:.1f}s{flag}")

        if args.patience > 0 and bad_epochs >= args.patience:
            print("Early stopping.")
            break

    # Load best & evaluate test
    if os.path.isfile(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))
        print(f"Loaded best weights from: {save_path}")

    if test_loader is not None:
        test_loss, test_acc = evaluate(model, test_loader, device, criterion)
        print(f"TEST  loss {test_loss:.4f}  acc {test_acc:.4f}")

if __name__ == "__main__":
    main()
