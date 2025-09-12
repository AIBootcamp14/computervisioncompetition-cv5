# 📑 Document Type Classification | 문서 타입 분류
## Team

| ![김수환](https://github.com/user-attachments/assets/bfe05d23-81d0-4409-aca9-b1bb1fb5107f) | ![김명철](https://github.com/user-attachments/assets/0c545d12-539f-419d-816a-a0e4263cc0b2) | ![김상윤](https://github.com/user-attachments/assets/5bd23640-3d34-4292-bc81-e202136a1b6f) | ![김광묵](https://github.com/user-attachments/assets/5aee2fa3-df3c-4183-a780-f2028ad613ca) | ![장윤정](https://github.com/user-attachments/assets/bee0c0c4-ae06-4477-8ea6-a3cdaf2b00f8) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [김수환](https://github.com/suhwankimkim)             |            [김명철](https://github.com/qpwpep)             |            [김상윤](https://github.com/94KSY)             |            [김광묵](https://github.com/JackFink)             |            [장윤정](https://github.com/yjjang06)             |
|                            팀장, <br>실험환경 구성 및 모델 개발                             |                            모델 개발                             |                            모델 개발                             |                            모델 개발                             |                            모델 개발                             |

## 0. Overview
### Environment
- Python 3.12

### Requirements
```
ipykernel==6.27.1
ipython==8.15.0
ipywidgets==8.1.1
jupyter==1.0.0
matplotlib==3.10.6
matplotlib-inline==0.1.6
numpy==1.26.0
pandas==2.1.4
Pillow==9.4.0
timm==0.9.12
torch==2.8.0
pytorch-lightning==2.5.5
scikit-learn==1.7.1
albumentations==1.3.1
augraphy==8.2.6
tqdm==4.67.1
hydra-core==1.3.2
wandb==0.21.3
```

## 1. Competiton Info

### Overview

- 문서는 금융, 보험, 물류, 의료 등 도메인을 가리지 않고 많이 취급됩니다. <br>이 대회는 다양한 종류의 문서 이미지의 클래스를 예측합니다.

### Timeline

- **프로젝트 기간**: 2025-09-01 - 2025-09-11
- **최종 제출**: 2025-09-11 (19시)

## 2. Components

### Directory
```
├── code
│   ├── notebooks
│   │   ├── baseline.ipynb
│   │   └── requirements.txt
│   └── train.py
├── src
│   ├── dataset
│   │   └── docs_image.py
│   ├── model
│   │   └── timmCls.py
│   ├── util
│   │   ├── transform.py
│   │   └── utils.py
│   ├── train.py
│   └── requirements.txt
├── data
│   ├── train
│   ├── test
│   ├── train.csv
│   ├── meta.csv
│   └── submission.csv
```

## 3. Data descrption

### Dataset overview

- data/train, data/train.csv
    - 1570장의 이미지, train/ 폴더에 존재하는 1570장의 이미지에 대한 정답 클래스 csv
- data/test, data/sample_submission.csv
    - 3140장의 이미지, test/ 폴더에 존재하는 3140장의 이미지에 대한 target 예측 결과가 0으로 저장된 샘플 제출용 csv
- data/meta.csv
    - target별 class_name


### EDA

- **이미지 데이터에 대한 분포 확인**
    - 이미지 크기, 비율, 색상 분포 확인
    
- **test set 이미지에 적용된 증강 확인**
    - 회전, 플립, 크롭, 색상 변환 등 다양한 증강이 적용되어 있음을 확인


### Data Processing

- **Augmentations**
    - test set 이미지와 비슷하게 train set 이미지에도 **강한 증강**을 적용
    - 온라인 증강과 오프라인 증강의 장단점을 팀원들과 논의하여 **온라인 증강**을 적용하기로 함
    - **ReplayCompose**를 적용하여 동일한 증강이 적용될 수 있도록 함
    - 크롭, 리사이즈, 패딩 등을 적용 시, 여백 배경색에 대한 고민, 흰색, 검정색, 이미지 색상값의 중간값 등 여러 방법으로 적용하여 실험

## 4. Modeling

### Model descrition

- 'convnextv2_large.fcmae_ft_in22k_in1k_384'
- 'efficientnetv2_rw_m.agc_in1k'
- 'regnety_320.swag_ft_in1k'
- 'deit3_large_patch16_384.fb_in22k_ft_in1k'
- 'swinv2_base_window12to24_192to384.ms_in22k_ft_in1k'

적은 데이터 수로 인해 CNN 기반의 모델(convnextv2, efficientnetv2, regnet)을 주로 사용.
다만 적은 데이터 셋이라도 트랜스포머 기반의 모델(deit3, swinv2) 역시 좋은 성능을 보여줬기 때문에 VIT/SWIN 계열의 모델 2종을 추가하여 앙상블을 진행.  

### Modeling Process

1. 5개의 모델을 StratifiedKFold를 이용해 학습.

2. Fold마다 softmax로 로짓 값을 변환후,각 Fold의 확률을 평균하여 모델의 평균을 계산.

3. 각 모델의 평균을 모델별 가중치와 곱해 가중 평균을 구하고, argmax를 취해 최종 평균을 산출.


## 5. Result

### Leader Board

<img src="https://github.com/user-attachments/assets/19ffe206-cd53-499e-a7a6-18c714cbe809" />

- **Evaluation Metric**: Macro F1 Score
- **Leader Board Score**: 0.9517 (4위)

### Presentation

- [AI-부트캠프-14기_CV-경진대회-발표(5조).pptx](https://github.com/AIBootcamp14/computervisioncompetition-cv5/blob/666280248cc25273f07c277fee4373f0cea4b6ec/docs/%5B%E1%84%8F%E1%85%A5%E1%84%82%E1%85%A5%E1%86%AF%E1%84%8B%E1%85%A1%E1%84%8F%E1%85%A1%E1%84%83%E1%85%A6%E1%84%86%E1%85%B5%5D-AI-%E1%84%87%E1%85%AE%E1%84%90%E1%85%B3%E1%84%8F%E1%85%A2%E1%86%B7%E1%84%91%E1%85%B3-14%E1%84%80%E1%85%B5_CV-%E1%84%80%E1%85%A7%E1%86%BC%E1%84%8C%E1%85%B5%E1%86%AB%E1%84%83%E1%85%A2%E1%84%92%E1%85%AC-%E1%84%87%E1%85%A1%E1%86%AF%E1%84%91%E1%85%AD(5%E1%84%8C%E1%85%A9).pdf)

## etc

### Meeting Log
1~2일 간격으로 주기적인 미팅으로 진행상황 및 이슈사항 공유

|날짜|회의 내용|
|-----|-----|
|2025-08-27|• Team Building|
|2025-09-01|• 대회 개요 및 데이터 구조 파악, 평가 지표 이해를 위한 계산식 확인<br>• 서버 생성 및 환경 구축, Baseline실행 및 submission 파일 제출<br>• train, test set에 대한 EDA 진행|
|2025-09-03|• EDA 진행 내용 공유<br>• Hydra, wandb 등 실험 환경 구축에 대한 내용 논의|
|2025-09-05|• 각자 실험을 위한 코드 개발 및 실험 진행 내용 공유<br>• 증강 방법에 대한 내용 논의|
|2025-09-09|• 각자 진행한 실험 내용 공유<br>• validation set 증강 재현에 대한 내용 논의<br>• Hydra, wandb 사용하여 실험 환경 작업 내용 공유|
|2025-09-10|• 각자 진행한 실험 내용 공유|
|2025-09-11|• 각자 진행한 실험 내용 공유|

### Reference
- [\[2301.00808\] ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders](https://arxiv.org/abs/2301.00808)
- [\[2104.00298\] EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298)
- [\[2204.07118\] DeiT III: Revenge of the ViT](https://arxiv.org/abs/2204.07118)
- [\[2101.00590\] RegNet: Self-Regulated Network for Image Classification](https://arxiv.org/abs/2101.00590)
- [\[2111.09883\] Swin Transformer V2: Scaling Up Capacity and Resolution](https://arxiv.org/abs/2111.09883)
#### Libraries
- [timm](https://github.com/huggingface/pytorch-image-models)
- [Albumentations](https://albumentations.ai/docs/)
- [Augraphy](https://augraphy.readthedocs.io/en/latest/)
- [Hydra](https://hydra.cc/docs/intro/)
