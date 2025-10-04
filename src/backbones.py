import torch


def build_backbone(name: str, pretrained: bool, global_pool: str, device: torch.device, input_size: int):
    try:
        import timm
    except ImportError as e:
        raise ImportError("Please install timm: pip install timm") from e


    try:
        m = timm.create_model(name, pretrained=pretrained, num_classes=0, global_pool=global_pool)
    except Exception as ex:
        m = timm.create_model(name, pretrained=False, num_classes=0, global_pool=global_pool)
        print(f"[WARN] Pretrained not loaded for {name}: {ex}")


    feat_dim = getattr(m, 'num_features', None)
    if feat_dim is None:
        m = m.to(device).eval()
        with torch.no_grad():
            x = torch.zeros(1,3,input_size,input_size, device=device)
            f = m(x)
            if f.ndim == 4:
                f = torch.nn.functional.adaptive_avg_pool2d(f,1).flatten(1)
            feat_dim = f.shape[1]
        m = m.cpu()
    return m, int(feat_dim)