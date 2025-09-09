import albumentations as A
from albumentations.pytorch import ToTensorV2

def build_train_tf(img_size=384):
    return A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(img_size, img_size, border_mode=0, value=0),
        A.OneOf([
            A.MotionBlur(p=0.3),
            A.GaussianBlur(p=0.3),
            A.MedianBlur(blur_limit=5, p=0.4),
        ], p=0.6),
        A.OneOf([
            A.GaussNoise(var_limit=(5.0, 40.0), p=0.5),
            A.ISONoise(p=0.2),
        ], p=0.5),
        A.ImageCompression(quality_lower=40, quality_upper=80, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.04, scale_limit=0.1, rotate_limit=7, border_mode=0, value=0, p=0.7),
        A.RandomBrightnessContrast(0.15, 0.1, p=0.6),
        A.CLAHE(p=0.2),
        A.HorizontalFlip(p=0.5),
        A.Normalize(),
        ToTensorV2(),
        ])



def build_val_tf(img_size=384):
    return A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(img_size, img_size, border_mode=0, value=0),
        A.GaussianBlur(p=0.2),
        A.ImageCompression(quality_lower=60, quality_upper=95, p=0.3),
        A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.03, rotate_limit=3, border_mode=0, value=0, p=0.4),
        A.Normalize(),
        ToTensorV2(),
        ])


def build_test_tf(img_size=384):
    return A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(img_size, img_size, border_mode=0, value=0),
        A.Normalize(),
        ToTensorV2(),
        ])

def build_tta_list(img_size=384):
    base = [
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(img_size, img_size, border_mode=0, value=0),
    ]
    tta_list = [
        A.Compose([*base, A.Normalize(), ToTensorV2()]),
        A.Compose([*base, A.HorizontalFlip(p=1.0), A.Normalize(), ToTensorV2()]),
        A.Compose([*base, A.Rotate(limit=5, border_mode=0, value=0, p=1.0), A.Normalize(), ToTensorV2()]),
        A.Compose([*base, A.GaussianBlur(p=1.0), A.Normalize(), ToTensorV2()]),
    ]
    return tta_list