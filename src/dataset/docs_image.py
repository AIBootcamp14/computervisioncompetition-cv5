import os, sys
import pickle
from typing import Optional, Dict, Any, Tuple

sys.path.append(os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
))

import pandas as pd
import numpy as np
import albumentations as A
from PIL import Image 
from torch.utils.data import Dataset

from src.utils.transform import build_replay_tf

class ImageDataset(Dataset):
    def __init__(self, csv, path, transform = None):
        if isinstance(csv, (str, os.PathLike)):
            self.df = pd.read_csv(csv).values
        elif isinstance(csv, pd.DataFrame):
            self.df = csv.values
        self.path = path
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        name, target = self.df[idx]
        img = np.array(Image.open(os.path.join(self.path, name)))
        if self.transform:
            img = self.transform(image=img)['image']
        return img, target
    

class ReplayValImageDataset(Dataset):
    """
    CSV: [filename, label] 형식 (네 기존 ImageDataset과 동일)
    첫 에폭: 확률적 변형을 적용하고, 샘플별 replay dict를 캐시에 저장
    이후 에폭: 저장한 replay dict로 동일 변형을 재현

    Parameters
    ----------
    csv : Union[str, os.PathLike, pd.DataFrame]
        파일 경로나 DataFrame. 2컬럼 [name, target] 가정.
    path : str
        이미지 루트 디렉토리
    replay_tf : A.ReplayCompose
        첫 에폭에 사용할 Albumentations ReplayCompose 파이프라인
    cache_dir : Optional[str]
        리플레이 캐시를 저장할 디렉토리. 지정하면 워커/런 재시작에도 유지됨
    cache_name : str
        캐시 파일명 (동시에 여러 설정을 구분하고 싶을 때 바꿔 쓰기)
    """

    def __init__(self, csv, path: str, replay_tf: Optional[A.ReplayCompose] = None, cache_dir: Optional[str] = None,
        cache_name: str = "val_replay.pkl",):

        if isinstance(csv, (str, os.PathLike)):
            self.df = pd.read_csv(csv).values
        elif isinstance(csv, pd.DataFrame):
            self.df = csv.values
        else:
            raise ValueError("csv must be a path or a pandas.DataFrame")

        self.path = path
        self.replay_tf = replay_tf or build_replay_tf()

        # 캐시 준비
        self.cache_dir = cache_dir
        self.cache_path = None
        if self.cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
            self.cache_path = os.path.join(cache_dir, cache_name)

        self._cache = {}
        if self.cache_path and os.path.exists(self.cache_path):
            with open(self.cache_path, "rb") as f:
                self._cache = pickle.load(f)

    def __len__(self):
        return len(self.df)

    def _save_cache(self):
        if self.cache_path:
            # 매 샘플 저장 방식: 멀티워커에서도 안전하게 누적되도록 자주 덮어쓴다
            with open(self.cache_path, "wb") as f:
                pickle.dump(self._cache, f)

    def __getitem__(self, idx):
        name, target = self.df[idx]
        img_path = os.path.join(self.path, name)

        # PIL 로드 후 np.uint8 RGB 보장
        img = np.array(Image.open(img_path))

        key = str(name)  # 파일명을 안정 키로 사용 (폴더 구조 포함이면 상대경로/절대경로로 통일해도 OK)

        if key in self._cache:
            # 저장된 리플레이로 동일 변형 재현
            out = A.ReplayCompose.replay(self._cache[key], image=img)
            img = out["image"]
        else:
            # 첫 에폭: 랜덤 변형 적용 + replay 저장
            out = self.replay_tf(image=img)
            img = out["image"]
            self._cache[key] = out["replay"]
            self._save_cache()

        return img, target
