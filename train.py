# ================= Notebook-ready training script =================
import os, math, time, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ---------- 1) PARAMS (đổi trực tiếp ở đây) ----------
DATASET_ROOT   = "Dataset"         # có train/val/test mỗi cái gồm fake/real
DATASET_ALIAS  = Path(DATASET_ROOT).name      # dùng để đặt tên file lưu (đổi tùy ý)
MODEL_KEY      = "efficientnet_b3" # {"efficientnet_b3", "xception"}
INPUT_SIZE     = 224               # 224 cho cả 2 model (ok); Xception gốc 299 vẫn chạy 224
BATCH_SIZE     = 32
EPOCHS         = 10
LR             = 1e-4
WEIGHT_DECAY   = 1e-4
NUM_WORKERS    = 6                 # Tăng workers để load data nhanh hơn
USE_AMP        = True              # mixed precision (CUDA)
SEED           = 42
PREFETCH_FACTOR = 2                # Prefetch batches để giảm GPU idle time
DROP_CONNECT   = 0.2               # cho EfficientNetB3
DROPOUT        = 0.3               # cho EfficientNetB3
CHECKPOINT_DIR = "checkpoints"

# ---------- 2) SETUP ----------
def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
set_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# GPU memory optimization
if device.type == "cuda":
    torch.cuda.empty_cache()
    print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ---------- 3) DATA ----------
# Chuẩn ImageNet
mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

train_tfms = transforms.Compose([
    transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
eval_tfms = transforms.Compose([
    transforms.Resize(int(INPUT_SIZE * 1.15)),
    transforms.CenterCrop(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

def make_loader(split, tfm, shuffle):
    p = os.path.join(DATASET_ROOT, split)
    ds = datasets.ImageFolder(p, transform=tfm)
    # Đảm bảo mapping nhãn ổn định (mong muốn: {'fake':0,'real':1})
    print(f"[{split}] classes ->", ds.classes, ds.class_to_idx)
    return ds, DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                          num_workers=NUM_WORKERS, pin_memory=True,
                          prefetch_factor=PREFETCH_FACTOR, persistent_workers=True)

train_ds, train_loader = make_loader("train", train_tfms, True)
val_ds,   val_loader   = make_loader("val",   eval_tfms,   False)
test_ds,  test_loader  = make_loader("test",  eval_tfms,   False)

NUM_CLASSES = len(train_ds.classes)

# ---------- 4) MODEL ----------
# Import từ file người dùng cung cấp
from efficientnet import EfficientNetB3
from xception import Xception

def build_model(key: str, num_classes: int):
    key = key.lower()
    if key == "efficientnet_b3":
        model = EfficientNetB3(num_classes=num_classes,
                               drop_connect_rate=DROP_CONNECT,
                               dropout=DROPOUT)
        name = "efficientnet_b3"
    elif key == "xception":
        model = Xception(num_classes=num_classes)
        name = "xception"
    else:
        raise ValueError(f"MODEL_KEY không hợp lệ: {key}")
    return model, name

model, model_name = build_model(MODEL_KEY, NUM_CLASSES)
model = model.to(device)

# PyTorch 2.0+ compile để tăng tốc
try:
    model = torch.compile(model, mode='reduce-overhead')
    print("✅ Model compiled with torch.compile")
except Exception as e:
    print(f"⚠️ torch.compile failed (PyTorch < 2.0?): {e}")

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs (DataParallel).")
    model = nn.DataParallel(model)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ---------- 5) TRAIN / EVAL LOOPS ----------
scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and device.type == "cuda"))

def run_epoch(loader, train_mode=True):
    model.train(train_mode)
    total, correct, loss_sum = 0, 0, 0.0
    progress_interval = max(1, len(loader) // 10)  # Show progress 10 times per epoch
    
    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.set_grad_enabled(train_mode):
            if scaler.is_enabled():
                with torch.cuda.amp.autocast():
                    logits = model(x)
                    loss = criterion(logits, y)
            else:
                logits = model(x)
                loss = criterion(logits, y)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        loss_sum += loss.item() * x.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += x.size(0)
        
        # Show progress during training
        if train_mode and (batch_idx + 1) % progress_interval == 0:
            current_acc = correct / total
            print(f"  📈 Batch {batch_idx+1}/{len(loader)} | "
                  f"Loss: {loss.item():.4f} | Acc: {current_acc:.4f}")
    
    return loss_sum / total, correct / total

# ---------- 6) TRAINING ----------
best_val = -1.0
ckpt_path = os.path.join(CHECKPOINT_DIR, f"{model_name}_{DATASET_ALIAS}_best.pth")

# Warmup: chạy vài batch đầu để "nóng máy" GPU
print("⚡ Warming up GPU...")
model.train()
warmup_batches = min(3, len(train_loader))
for i, (x, y) in enumerate(train_loader):
    if i >= warmup_batches:
        break
    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
    with torch.cuda.amp.autocast():
        _ = model(x)
print(f"✅ Warmup completed ({warmup_batches} batches)")

for ep in range(1, EPOCHS + 1):
    print(f"\n🚀 Starting Epoch {ep}/{EPOCHS}...")
    t0 = time.time()
    tr_loss, tr_acc = run_epoch(train_loader, True)
    val_loss, val_acc = run_epoch(val_loader,  False)
    scheduler.step()

    if val_acc > best_val:
        best_val = val_acc
        # Lưu duy nhất best (state_dict để gọn, tương thích)
        to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        torch.save({"model": to_save,
                    "epoch": ep,
                    "val_acc": best_val,
                    "model_name": model_name,
                    "dataset_alias": DATASET_ALIAS,
                    "input_size": INPUT_SIZE}, ckpt_path)

    print(f"Epoch {ep:02d} | "
          f"train_loss {tr_loss:.4f} acc {tr_acc:.4f} | "
          f"val_loss {val_loss:.4f} acc {val_acc:.4f} | "
          f"best_val {best_val:.4f} | time {time.time()-t0:.1f}s")

print(f"\nBest checkpoint saved to: {ckpt_path}")

# ---------- 7) TEST với best ----------
if os.path.isfile(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt["model"]
    (model.module if isinstance(model, nn.DataParallel) else model).load_state_dict(sd)

test_loss, test_acc = run_epoch(test_loader, False)
print(f"TEST: loss {test_loss:.4f} | acc {test_acc:.4f}")
