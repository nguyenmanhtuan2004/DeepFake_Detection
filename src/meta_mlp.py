import time, numpy as np, torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class MetaMLP(nn.Module):
    def __init__(self, in_dim: int, num_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
        nn.Linear(in_dim, 2048), nn.ReLU(inplace=True), nn.Dropout(dropout),
        nn.Linear(2048, 1024), nn.ReLU(inplace=True), nn.Dropout(dropout),
        nn.Linear(1024, 512), nn.ReLU(inplace=True), nn.Dropout(dropout),
        nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(dropout),
        nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(dropout),
        nn.Linear(128, num_classes),
        )
    def forward(self, x):
        return self.net(x)
    
def _make_subset_loader(X, y, batch_size, max_samples=None, seed=42, stratified=True, shuffle=True, device=None):
    n = len(X)
    if (max_samples is not None) and (max_samples < n):
        if stratified:
            from sklearn.model_selection import StratifiedShuffleSplit
            sss = StratifiedShuffleSplit(n_splits=1, train_size=max_samples, random_state=seed)
            idx, _ = next(sss.split(np.zeros(n), y))
        else:
            rng = np.random.RandomState(seed)
            idx = rng.choice(n, size=max_samples, replace=False)
        Xb, yb = X[idx], y[idx]
    else:
        Xb, yb = X, y


    Xb_t = torch.from_numpy(Xb).float().to(device, non_blocking=True)
    yb_t = torch.from_numpy(yb).long().to(device, non_blocking=True)
    ds = TensorDataset(Xb_t, yb_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)

def train_meta(Xtr, ytr, Xva, yva, save_path,
epochs, batch_size, lr, weight_decay, dropout,
epoch_sample, stratified_epoch_sample,
use_amp: bool, device: torch.device):
    torch.set_float32_matmul_precision("high")
    va_dl = _make_subset_loader(Xva, yva, batch_size, max_samples=None, shuffle=False, device=device)
    num_classes = int(len(np.unique(ytr)))
    model = MetaMLP(in_dim=Xtr.shape[1], num_classes=num_classes, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type=='cuda')
    best, best_sd = -1.0, None
    for ep in range(1, epochs+1):
        t0 = time.time()
        # dynamic epoch sampling: if very large train set, sample 20k per epoch
        n_train = len(Xtr)
        max_ep_samples = 20000 if n_train > 140000 else epoch_sample
        if ep == 1:
            print(f"[Meta] n_train={n_train} | cfg.epoch_sample={epoch_sample} -> using per-epoch sample = {max_ep_samples}")
        tr_dl = _make_subset_loader(
        Xtr, ytr, batch_size,
        max_samples=max_ep_samples, seed=42+ep,
        stratified=stratified_epoch_sample, shuffle=True, device=device
        )

        model.train(); tot=cor=0; loss_sum=0.0
        for xb, yb in tr_dl:
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp and device.type=='cuda'):
                logits = model(xb)
                loss = crit(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            loss_sum += float(loss.item()) * xb.size(0)
            cor += (logits.argmax(1)==yb).sum().item(); tot += xb.size(0)
        tr_loss, tr_acc = loss_sum/tot, cor/tot


        model.eval(); tot=cor=0; loss_sum=0.0
        with torch.no_grad():
            for xb, yb in va_dl:
                with torch.cuda.amp.autocast(enabled=use_amp and device.type=='cuda'):
                    logits = model(xb)
                    loss = crit(logits, yb)
                loss_sum += float(loss.item()) * xb.size(0)
                cor += (logits.argmax(1)==yb).sum().item(); tot += xb.size(0)
        va_loss, va_acc = loss_sum/tot, cor/tot
        if va_acc > best:
            best, best_sd = va_acc, {k: v.detach().cpu() for k, v in model.state_dict().items()}


        print(f"Meta Epoch {ep:02d} | train {tr_loss:.4f}/{tr_acc:.4f} | val {va_loss:.4f}/{va_acc:.4f} | best {best:.4f} | {time.time()-t0:.1f}s")


    torch.save({"state_dict": best_sd, "in_dim": Xtr.shape[1], "num_classes": num_classes}, save_path)
    print(f"Saved meta model: {save_path}")
    model.load_state_dict(best_sd)
    return model

def evaluate_meta(model, Xte, yte, batch_size: int, use_amp: bool, device: torch.device):
    dl = _make_subset_loader(Xte, yte, batch_size, max_samples=None, shuffle=False, device=device)
    model.eval(); tot=cor=0; loss_sum=0.0
    crit = nn.CrossEntropyLoss()
    with torch.no_grad():
        for xb, yb in dl:
            with torch.cuda.amp.autocast(enabled=use_amp and device.type=='cuda'):
                logits = model(xb); loss = crit(logits, yb)
            loss_sum += float(loss.item()) * xb.size(0)
            cor += (logits.argmax(1)==yb).sum().item(); tot += xb.size(0)
    print(f"TEST (meta): loss {loss_sum/tot:.4f} | acc {cor/tot:.4f}")
