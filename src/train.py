import os
import sys
sys.path.append(os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)))) # set working directory project


from dotenv import load_dotenv
import hydra
import pandas as pd
import numpy as np
import wandb

import torch
from torch.utils.data import DataLoader
from hydra.utils import instantiate
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from sklearn.model_selection import train_test_split
from omegaconf import OmegaConf

from src.dataset.docs_image import ImageDataset
from src.utils.utils import set_seed, project_path
from src.utils.transform import build_test_tf, build_train_tf, build_val_tf

# set seed
set_seed()

# define path
pj_path = project_path()
data_path = os.path.join(pj_path, 'data')
train_path = os.path.join(data_path, 'train')
test_path = os.path.join(data_path, 'test')

load_dotenv()


def _get_opt_name(cfg):
    target = OmegaConf.select(cfg, "optimizer._target_", default=None)
    return target.split(".")[-1].lower() if target else "opt"

def _build_run_name(cfg):
    import wandb
    backbone = OmegaConf.select(cfg, "model.model.backbone.model_name", default="model")  # 현재 트리 구조 기준
    opt_name = _get_opt_name(cfg)
    epochs   = OmegaConf.select(cfg, "train.trainer.max_epochs", default="??")
    fold     = OmegaConf.select(cfg, "fold", default=None)
    suffix   = f"-f{fold}" if fold is not None else ""
    short_id = wandb.util.generate_id()[:4]
    return f"{backbone}-{opt_name}-ep{epochs}{suffix}-{short_id}"

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




@hydra.main(config_path='../configs', config_name = 'config')
def main(cfg):
    wandb.login()
    # load dataframe by image path 
    train = pd.read_csv(os.path.join(data_path, 'train.csv'))
    test = pd.read_csv(os.path.join(data_path, 'sample_submission.csv'))
    
    train, valid = train_test_split(train, test_size= .2, random_state = 42, stratify=train['target'])

    # define transform
    trn_transform = build_train_tf()
    val_transform = build_val_tf()
    tst_transform = build_test_tf()

    # define data set and dataloader 
    train_dataset = ImageDataset(train, train_path, transform= trn_transform)
    valid_dataset = ImageDataset(valid, train_path, transform= val_transform)
    test_dataset = ImageDataset(test, test_path, transform= tst_transform)

    train_loader = DataLoader(train_dataset, batch_size= 32, shuffle = True)
    valid_loader = DataLoader(valid_dataset, batch_size = 32 ,shuffle = False)
    test_loader = DataLoader(test_dataset,batch_size= 32,shuffle = False)

    # model define
    model = instantiate(cfg.model.model, _recursive_ = False)


    # define callback
    ckpt = ModelCheckpoint(
        monitor = cfg.callback.monitor,
        mode = cfg.callback.mode, save_top_k= 1,
        filename="best-{epoch:02d}-{valid_f1:.4f}",
        auto_insert_metric_name=False
    )
    early_stopping = EarlyStopping(monitor = cfg.callback.monitor,
                                   mode = cfg.callback.mode, patience= cfg.callback.patience)
    lr_monitor = LearningRateMonitor(logging_interval=cfg.callback.logging_interval)

    run_name = _build_run_name(cfg)
    logger_params = OmegaConf.to_container(cfg.logger, resolve=False) or {}
    logger_params = dict(logger_params) if logger_params is not None else {}
    logger_params["name"] = run_name
    logger = WandbLogger(**logger_params)

    # define trainer
    trainer = Trainer(**cfg.train.trainer, callbacks = [ckpt ,early_stopping, lr_monitor], logger = logger)

    # model fitting
    trainer.fit(model, train_loader, valid_loader)

    # optional inference
    if OmegaConf.select(cfg, "infer.enabled", default=False):
    # 1) ckpt 경로 선택
        if OmegaConf.select(cfg, "infer.use_best_ckpt", default=True):
            ckpt_path = ckpt.best_model_path
            if not ckpt_path:
                raise FileNotFoundError("best_model_path가 비어있습니다. valid_f1 로깅을 확인하세요.")
        else:
            ckpt_path = OmegaConf.select(cfg, "infer.ckpt_path", default=None)
            if not ckpt_path or not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"ckpt_path가 존재하지 않습니다: {ckpt_path}")

        # 2) LightningModule 복원
        PredModel = type(model)
        pred_model = PredModel.load_from_checkpoint(ckpt_path)
        pred_model.eval()

        # 3) Trainer.predict → logits → argmax
        raw_outputs = trainer.predict(pred_model, dataloaders=test_loader)
        preds = torch.cat([out.argmax(1).cpu() for out in raw_outputs], 0).numpy()

        # 4) submission 저장 (run_name 재사용)
        sample_csv = os.path.join(data_path, "sample_submission.csv")
        sub = pd.read_csv(sample_csv)
        if len(sub) != len(preds):
            raise ValueError(f"submission rows({len(sub)}) != preds({len(preds)})")
        sub["target"] = preds

        out_dir = os.path.join(pj_path, cfg.infer.submission_dir)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{cfg.infer.filename_prefix}-{run_name}.csv")
        sub.to_csv(out_path, index=False)
        print(f"[SUBMISSION SAVED] {out_path}")



if __name__ == '__main__':
    main()