import torch, numpy as np


@torch.no_grad()
def extract_stacked_features(backbones, loader, device: torch.device, use_amp: bool):
    for m in backbones: m.to(device).eval()
    feats, labels = [], []


    amp_ctx = torch.cuda.amp.autocast(enabled=use_amp and device.type=='cuda')


    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with amp_ctx:
            f_list = []
            for m in backbones:
                f = m(x)
                if f.ndim == 4:
                    f = torch.nn.functional.adaptive_avg_pool2d(f,1).flatten(1)
                f_list.append(f)
            f_cat = torch.cat(f_list, dim=1)
        feats.append(f_cat.cpu().float())
        labels.append(y.clone())


    F = torch.cat(feats, 0).numpy()
    y = torch.cat(labels, 0).numpy()


    for m in backbones: m.cpu()
    return F, y