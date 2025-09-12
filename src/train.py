import os
import sys
sys.path.append(os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)))) # set working directory project
from pathlib import Path
from datetime import datetime

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

from src.dataset.docs_image import ImageDataset, ReplayValImageDataset
from src.models.timmCls import TimmClassifier
from src.utils.utils import *
from src.utils.transform import build_test_tf, build_train_tf, build_val_tf



# set seed
set_seed()

# define path
pj_path = project_path()
data_path = os.path.join(pj_path, 'data')
train_path = os.path.join(data_path, 'train')
test_path = os.path.join(data_path, 'test')

now = datetime.now().strftime("%mm%dd%H%M")

load_dotenv()

@hydra.main(config_path='../configs', config_name = 'config')
def main(cfg):
    wandb.login()
    # load dataframe by image path 
    train = pd.read_csv(os.path.join(data_path, 'train.csv'))
    test = pd.read_csv(os.path.join(data_path, 'sample_submission.csv'))
    if not cfg['cv_ensure'] :
        train, valid = train_test_split(train, test_size= .2, random_state = cfg.data.seed, stratify=train['target'])

        # define transform
        trn_transform = build_train_tf(img_size=cfg.data.image_size)
        val_transform = build_val_tf(img_size=cfg.data.image_size)
        tst_transform = build_test_tf(img_size=cfg.data.image_size)

        # define data set and dataloader 
        train_dataset = ImageDataset(train, train_path, transform= trn_transform)
        if cfg.data.replay:
            valid_dataset = ReplayValImageDataset(valid, train_path, replay_tf = None, cache_dir=os.path.join(pj_path, f'cache/{now}'), cache_name=f"val_replay.pkl")
        else:
            valid_dataset = ImageDataset(valid, train_path, transform= val_transform)
        test_dataset = ImageDataset(test, test_path, transform= tst_transform)

        train_loader = DataLoader(train_dataset, batch_size = cfg.data.batch_size, shuffle = True)
        valid_loader = DataLoader(valid_dataset, batch_size = cfg.data.batch_size, shuffle = False)
        test_loader = DataLoader(test_dataset, batch_size = cfg.data.batch_size, shuffle = False)

        # model define
        model = TimmClassifier(num_classes = cfg.model.model.num_classes, 
                            backbone = cfg.model.model.backbone,
                            optimizer_cfg = cfg.optimizer, 
                            scheduler_cfg = OmegaConf.select(cfg, 'scheduler', default=None))


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

        run_name = build_run_name(cfg)
        logger_params = OmegaConf.to_container(cfg.logger, resolve=True) or {}
        logger_params = dict(logger_params) if logger_params is not None else {}
        logger_params["name"] = run_name + f'[hold_out]'
        logger = WandbLogger(**logger_params)

        # define trainer
        trainer = Trainer(**cfg.train.trainer, callbacks = [ckpt ,early_stopping, lr_monitor], logger = logger)

        # model fitting
        trainer.fit(model, train_loader, valid_loader)

        # optional inference
        if OmegaConf.select(cfg, "infer.infer.enabled", default=False):
        # 1) ckpt 경로 선택
            if OmegaConf.select(cfg, "infer.infer.use_best_ckpt", default=True):
                ckpt_path = ckpt.best_model_path
                if not ckpt_path:
                    raise FileNotFoundError("best_model_path가 비어있습니다. valid_f1 로깅을 확인하세요.")
                ckpt_path = str(Path(ckpt_path).resolve()) # 절대 경로로 변경
            else:
                ckpt_path = OmegaConf.select(cfg, "infer.infer.ckpt_path", default=None)
                if not ckpt_path or not os.path.exists(ckpt_path):
                    raise FileNotFoundError(f"ckpt_path가 존재하지 않습니다: {ckpt_path}")

            # 2) LightningModule 복원
            pred_model = TimmClassifier.load_from_checkpoint(ckpt_path)
            
            pred_model.eval()
            pred_trainer = Trainer(**cfg.train.trainer, logger=False, callbacks=[])
            # 3) Trainer.predict → logits → argmax
            raw_outputs = pred_trainer.predict(pred_model, dataloaders=test_loader)
            preds = torch.cat([out.argmax(1).cpu() for out in raw_outputs], 0).numpy()

            # 4) submission 저장 (run_name 재사용)
            sample_csv = os.path.join(data_path, "sample_submission.csv")
            sub = pd.read_csv(sample_csv)
            if len(sub) != len(preds):
                raise ValueError(f"submission rows({len(sub)}) != preds({len(preds)})")
            sub["target"] = preds

            out_dir = os.path.join(pj_path, cfg.infer.infer.submission_dir)
            os.makedirs(out_dir, exist_ok=True)

            fname = build_submission_name(cfg, mode="holdout")
            out_path = os.path.join(out_dir, fname)
            sub.to_csv(out_path, index=False)
            print(f"[SUBMISSION SAVED] {out_path}")
        logger.finalize('success')
    else:
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.data.seed)

        trn_transform = build_train_tf(img_size=cfg.data.image_size)
        val_transform = build_val_tf(img_size=cfg.data.image_size)   # replay 안 쓰면 사용
        tst_transform = build_test_tf(img_size=cfg.data.image_size)

        # 테스트 로더는 공용으로 1회만 생성
        test_dataset = ImageDataset(test, test_path, transform=tst_transform)
        test_loader  = DataLoader(test_dataset, batch_size=cfg.data.batch_size, shuffle=False)

        # fold별 확률을 담을 리스트 (확률 평균 앙상블)
        fold_probs = []   # each: [N_test, C]

        for fold, (train_idx, valid_idx) in enumerate(skf.split(train, y=train['target'])):
            print(f"\n============================== Fold {fold + 1}/5 ==============================")
            cfg.fold = fold + 1
            print(f'fold : {cfg.fold}')
            train_df = train.iloc[train_idx].reset_index(drop=True)
            valid_df = train.iloc[valid_idx].reset_index(drop=True)

            # datasets
            train_dataset = ImageDataset(train_df, train_path, transform=trn_transform)
            if not cfg.data.replay:
                valid_dataset = ImageDataset(valid_df, train_path, transform=val_transform)
            else:
                # 폴드별 캐시 디렉토리 분리 (충돌 방지)
                valid_dataset = ReplayValImageDataset(
                    valid_df, train_path,
                    replay_tf=None,  # 내부 build_replay_tf() 사용
                    cache_dir=os.path.join(pj_path, f'cache/{now}'),
                    cache_name=f'fold{fold + 1}_val_replay.pkl'
                )

            train_loader = DataLoader(train_dataset, batch_size=cfg.data.batch_size, shuffle=True)
            # 첫 에폭 캐시 생성 안정화를 위해 valid는 num_workers=0 권장
            valid_loader = DataLoader(valid_dataset, batch_size=cfg.data.batch_size, shuffle=False, num_workers=0)

            # model
            model = TimmClassifier(
                num_classes=cfg.model.model.num_classes,
                backbone=cfg.model.model.backbone,
                optimizer_cfg=cfg.optimizer,
                scheduler_cfg=OmegaConf.select(cfg, 'scheduler', default=None)
            )

            # callbacks
            ckpt = ModelCheckpoint(
                monitor=cfg.callback.monitor,
                mode=cfg.callback.mode, save_top_k=1,
                filename="best-{epoch:02d}-{valid_f1:.4f}",
                auto_insert_metric_name=False
            )
            early_stopping = EarlyStopping(monitor=cfg.callback.monitor, mode=cfg.callback.mode, patience=cfg.callback.patience)
            lr_monitor = LearningRateMonitor(logging_interval=cfg.callback.logging_interval)

            run_name = build_run_name(cfg) + f'fold-{fold+1}'            # 내부에 -f{fold} 포함됨
            group    = f"{cfg.model.model.backbone}-{get_opt_name(cfg)}"
            logger_params = OmegaConf.to_container(cfg.logger, resolve=False) or {}
            logger_params = dict(logger_params)

            logger_params.update({
                "name": run_name,             # 예: convnext_tiny-adamw-ep20-f3-1a2b
                "group": group,               # 같은 실험 묶음으로 보기 좋음
                "id": f"{wandb.util.generate_id()}-f{fold+1}",  # ★ 폴드마다 고유 ID
                "resume": "never",
                "tags": [f"fold={fold+1}", "cv"]
            })

            logger = WandbLogger(**logger_params)
            trainer = Trainer(**cfg.train.trainer, callbacks=[ckpt, early_stopping, lr_monitor], logger=logger)

            # fit
            trainer.fit(model, train_loader, valid_loader)

            # ---- fold별 추론 (확률 산출) ----
            if OmegaConf.select(cfg, "infer.infer.enabled", default=False):
                # 1) best ckpt 경로
                if OmegaConf.select(cfg, "infer.infer.use_best_ckpt", default=True):
                    ckpt_path = ckpt.best_model_path
                    if not ckpt_path:
                        raise FileNotFoundError("best_model_path가 비어있습니다. valid_f1 로깅을 확인하세요.")
                    ckpt_path = str(Path(ckpt_path).resolve())
                else:
                    ckpt_path = OmegaConf.select(cfg, "infer.infer.ckpt_path", default=None)
                    if not ckpt_path or not os.path.exists(ckpt_path):
                        raise FileNotFoundError(f"ckpt_path가 존재하지 않습니다: {ckpt_path}")

                # 2) LightningModule 복원
                pred_model = TimmClassifier.load_from_checkpoint(ckpt_path)
            
                pred_model.eval()
                pred_trainer = Trainer(**cfg.train.trainer, logger=False)
                raw_outputs = pred_trainer.predict(pred_model, dataloaders=test_loader)
                logits = torch.cat([out if out.ndim == 2 else out[0] for out in raw_outputs], dim=0)  # [N, C]
                probs  = torch.softmax(logits, dim=1)  # 멀티클래스 확률

                fold_probs.append(probs.cpu())
            logger.experiment.finish()
        # ---- 앙상블 & 제출 ----
        if OmegaConf.select(cfg, "infer.infer.enabled", default=False):
            # 확률 평균 앙상블
            ens_probs = torch.mean(torch.stack(fold_probs, dim=0), dim=0)  # [F,N,C] -> [N,C]
            preds = ens_probs.argmax(dim=1).numpy()

            # submission
            sample_csv = os.path.join(data_path, "sample_submission.csv")
            sub = pd.read_csv(sample_csv)
            if len(sub) != len(preds):
                raise ValueError(f"submission rows({len(sub)}) != preds({len(preds)})")
            sub["target"] = preds

            out_dir = os.path.join(pj_path, cfg.infer.infer.submission_dir)
            os.makedirs(out_dir, exist_ok=True)

            fname = build_submission_name(cfg, mode="cv", n_folds=len(fold_probs))
            out_path = os.path.join(out_dir, fname)
            sub.to_csv(out_path, index=False)
            print(f"[SUBMISSION SAVED] {out_path}")



if __name__ == '__main__':
    main()