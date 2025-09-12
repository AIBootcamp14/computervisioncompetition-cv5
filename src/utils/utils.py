import os
import random

import numpy as np
import torch
from omegaconf import OmegaConf
import wandb


def set_seed(SEED:int = 42):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True

def project_path():
    return os.path.join(
        os.path.dirname(
            os.path.abspath(__file__)
        ),
        "..",
        ".."
    )

def get_opt_name(cfg):
    target = OmegaConf.select(cfg, "optimizer._target_", default=None)
    return target.split(".")[-1].lower() if target else "opt"

def build_run_name(cfg):
    backbone = OmegaConf.select(cfg, "model.model.backbone", default="model")  # 현재 트리 구조 기준
    opt_name = get_opt_name(cfg)
    fold     = OmegaConf.select(cfg, "fold", default=None)
    suffix   = f"-f{fold}" if fold is not None else ""
    id       = wandb.util.generate_id()[:4]
    return f"{backbone}-{opt_name}-{suffix}-{id}"

def _to_class_indices(batch_out):
    """
    LightningModule의 predict_step/forward가 logits을 반환해도, 
    이미 클래스 인덱스를 반환해도 모두 대응.
    - logits (B, C) -> argmax
    - preds  (B,)   -> 그대로
    - 리스트/튜플 -> 첫 항목 사용
    """
    if isinstance(batch_out, (list, tuple)) and len(batch_out) > 0:
        batch_out = batch_out[0]
    if isinstance(batch_out, torch.Tensor):
        if batch_out.ndim == 2:         # logits
            return batch_out.argmax(dim=1)
        elif batch_out.ndim == 1:       # already indices
            return batch_out
    raise ValueError("predict_step/forward 출력 형식이 예상과 다릅니다. (Tensor[B,C] 또는 Tensor[B])")

def _get_backbone_name(cfg):
    return OmegaConf.select(cfg, "model.model.backbone", default="model")

def build_submission_name(cfg, *, mode: str, n_folds: int | None = None):
    backbone = _get_backbone_name(cfg)
    mode = mode.lower()
    kf = f"-kf{n_folds}" if (mode == "cv" and n_folds) else ""
    short_id = wandb.util.generate_id()[:4]
    opt = get_opt_name(cfg)
    return f"sub-{backbone}-{opt}-{mode}{kf}-{short_id}.csv"
