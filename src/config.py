from dataclasses import dataclass
from typing import List


@dataclass
class _Project:
    name: str
    seed: int
    out_dir: str
    save_path: str
@dataclass
class _Device:
    use_amp: bool
    prefer_gpu: bool

@dataclass
class _Normalize:
    mean: list
    std: list


@dataclass
class _Data:
    dataset_root: str
    input_size: int
    batch_size_img: int
    num_workers: int
    normalize: _Normalize


@dataclass
class _Backbone:
    name: str
    pretrained: bool
    global_pool: str
@dataclass
class _Rankers:
    xgboost: dict
    random_forest: dict


@dataclass
class _FS:
    keep_frac: float
    subsample: int
    rankers: _Rankers


@dataclass
class _MetaMLP:
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    dropout: float
    epoch_sample: int | None
    stratified_epoch_sample: bool

@dataclass
class _Logging:
    print_every: int
    save_best_only: bool
    eval_on_test_after_train: bool


@dataclass
class Cfg:
    project: _Project
    device: _Device
    data: _Data
    backbones: List[_Backbone]
    feature_selection: _FS
    meta_mlp: _MetaMLP
    logging: _Logging

def _to_dataclass(d: dict) -> Cfg:
    # Convert nested dict to dataclasses
    proj = _Project(**d['project'])
    dev = _Device(**d['device'])
    norm = _Normalize(**d['data']['normalize'])
    dat = _Data(normalize=norm, **{k:v for k,v in d['data'].items() if k!='normalize'})
    backs = [_Backbone(**b) for b in d['backbones']]
    rank = _Rankers(**d['feature_selection']['rankers'])
    fs = _FS(rankers=rank, **{k:v for k,v in d['feature_selection'].items() if k!='rankers'})
    mm = _MetaMLP(**d['meta_mlp'])
    log = _Logging(**d['logging'])
    return Cfg(project=proj, device=dev, data=dat, backbones=backs, feature_selection=fs, meta_mlp=mm, logging=log)




def load_config(path: str) -> Cfg:
    try:
        import yaml
    except ImportError as e:
        raise ImportError("Please install pyyaml: pip install pyyaml") from e


    with open(path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f)
    return _to_dataclass(raw)