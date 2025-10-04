import os, numpy as np
from pathlib import Path
from config import load_config
from utils import set_seed, get_device, ensure_dir
from data import make_loaders
from backbones import build_backbone
from extract_features import extract_stacked_features
from feature_selection import feature_selection_topk
from meta_mlp import train_meta, evaluate_meta
if __name__ == "__main__":
    cfg = load_config("config.yaml")
    set_seed(cfg.project.seed)


    out_dir = Path(cfg.project.out_dir)
    ensure_dir(out_dir)


    device = get_device(cfg.device.prefer_gpu)
    print(f"[Device] {device}")
    # --- DATALOADERS ---
    tr_ds, tr_dl, va_ds, va_dl, te_ds, te_dl = make_loaders(
    root=cfg.data.dataset_root,
    input_size=cfg.data.input_size,
    batch_size=cfg.data.batch_size_img,
    num_workers=min(cfg.data.num_workers, os.cpu_count() or 4),
    mean=cfg.data.normalize.mean,
    std=cfg.data.normalize.std,
    use_amp=cfg.device.use_amp,
    )
    num_classes = len(tr_ds.classes)
    print(f"[Data] Classes: {tr_ds.classes} ({num_classes})")
    # --- BACKBONES ---
    backs = []
    dims = []
    for bcfg in cfg.backbones:
        m, d = build_backbone(
        bcfg.name,
        pretrained=bcfg.pretrained,
        global_pool=bcfg.global_pool,
        device=device,
        input_size=cfg.data.input_size,
        )
        backs.append(m); dims.append(d)
        print(f"[Backbone] {bcfg.name} -> dim {d}")
    print(f"[Backbone] Stacked dim = {sum(dims)}")
    # --- FEATURE EXTRACTION (with caching) ---
    f_tr_path = out_dir / "F_tr.npy"; y_tr_path = out_dir / "y_tr.npy"
    f_va_path = out_dir / "F_va.npy"; y_va_path = out_dir / "y_va.npy"
    f_te_path = out_dir / "F_te.npy"; y_te_path = out_dir / "y_te.npy"
    if not f_tr_path.exists():
        F_tr, y_tr = extract_stacked_features(backs, tr_dl, device=device, use_amp=cfg.device.use_amp)
        np.save(f_tr_path, F_tr); np.save(y_tr_path, y_tr)
    else:
     F_tr = np.load(f_tr_path); y_tr = np.load(y_tr_path)


    if not f_va_path.exists():
        F_va, y_va = extract_stacked_features(backs, va_dl, device=device, use_amp=cfg.device.use_amp)
        np.save(f_va_path, F_va); np.save(y_va_path, y_va)
    else:
        F_va = np.load(f_va_path); y_va = np.load(y_va_path)


    if not f_te_path.exists():
        F_te, y_te = extract_stacked_features(backs, te_dl, device=device, use_amp=cfg.device.use_amp)
        np.save(f_te_path, F_te); np.save(y_te_path, y_te)
    else:
        F_te = np.load(f_te_path); y_te = np.load(y_te_path)


    print(f"[Feat] Train {F_tr.shape}, Val {F_va.shape}, Test {F_te.shape}")
    # --- FEATURE SELECTION ---
    keep_idx = feature_selection_topk(
    np.concatenate([F_tr, F_va], 0),
    np.concatenate([y_tr, y_va], 0),
    keep_frac=cfg.feature_selection.keep_frac,
    subsample=cfg.feature_selection.subsample,
    rank_cfg=cfg.feature_selection.rankers,
    )
    np.save(out_dir / "keep_idx.npy", keep_idx)
    print(f"[FS] Selected {len(keep_idx)} / {F_tr.shape[1]} ({cfg.feature_selection.keep_frac*100:.1f}%)")


    F_tr_sel = F_tr[:, keep_idx]; F_va_sel = F_va[:, keep_idx]; F_te_sel = F_te[:, keep_idx]

    # --- META MLP ---
    # Dynamic per-epoch sampling rule: if n_train > 140k -> sample 20k per epoch; else keep config
    n_train = F_tr_sel.shape[0]
    effective_epoch_sample = 20000 if n_train > 140000 else cfg.meta_mlp.epoch_sample
    if n_train > 140000:
        print(f"[Meta] n_train={n_train} > 140000 → override epoch_sample=20000 per epoch")
    else:
        print(f"[Meta] n_train={n_train} ≤ 140000 → use config epoch_sample={effective_epoch_sample}")


    save_path = str(Path(cfg.project.save_path))
    meta = train_meta(
    Xtr=F_tr_sel, ytr=y_tr,
    Xva=F_va_sel, yva=y_va,
    save_path=save_path,
    epochs=cfg.meta_mlp.epochs,
    batch_size=cfg.meta_mlp.batch_size,
    lr=cfg.meta_mlp.lr,
    weight_decay=cfg.meta_mlp.weight_decay,
    dropout=cfg.meta_mlp.dropout,
    epoch_sample=effective_epoch_sample, # <— apply dynamic rule here
    stratified_epoch_sample=cfg.meta_mlp.stratified_epoch_sample,
    use_amp=cfg.device.use_amp,
    device=device,
    )


    # --- EVAL ---
    evaluate_meta(meta, F_te_sel, y_te, batch_size=cfg.meta_mlp.batch_size, use_amp=cfg.device.use_amp, device=device)