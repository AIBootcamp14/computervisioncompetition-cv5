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

- September 1, 2025 ~ September 9, 2025

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

- _Describe your EDA process and step-by-step conclusion_

### Data Processing

- _Describe data processing process (e.g. Data Labeling, Data Cleaning..)_

## 4. Modeling

### Model descrition

- _Write model information and why your select this model_

### Modeling Process

- _Write model train and test process with capture_

## 5. Result

### Leader Board

<img src="https://github.com/user-attachments/assets/19ffe206-cd53-499e-a7a6-18c714cbe809" />

- **Evaluation Metric**: Macro F1 Score
- **Leader Board Score**: 0.9517 (4위)

### Presentation

- [AI-부트캠프-14기_CV-경진대회-발표(5조).pptx](https://docs.google.com/presentation/d/1lxRDSG-r6BOnA57wjGmnYVT071pgzeYd/edit?usp=sharing&ouid=101398214368344224612&rtpof=true&sd=true)

## etc

### Meeting Log

- 1~2일 간격으로 주기적인 미팅으로 진행상황 및 이슈사항 공유

### Reference

- _Insert related reference_
