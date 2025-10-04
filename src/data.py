import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode




def make_loaders(root: str, input_size: int, batch_size: int, num_workers: int, mean, std, use_amp: bool):
    tf = transforms.Compose([
    transforms.Resize(int(input_size*1.15), interpolation=InterpolationMode.BILINEAR),
    transforms.CenterCrop(input_size),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
    ])


    def _mk(split, shuffle):
        ds = datasets.ImageFolder(os.path.join(root, split), transform=tf)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=use_amp, persistent_workers=num_workers>0)
        return ds, dl


    tr_ds, tr_dl = _mk('train', True)
    va_ds, va_dl = _mk('val', False)
    te_ds, te_dl = _mk('test', False)
    return tr_ds, tr_dl, va_ds, va_dl, te_ds, te_dl