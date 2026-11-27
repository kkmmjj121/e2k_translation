# E2K Translation

AI Hub 영한 번역 데이터를 이용한 전처리, 모델 학습, 번역 평가 실험 기록입니다. 사전학습 번역 모델의 fine-tuning과 직접 구현한 Transformer의 scratch 학습 코드를 함께 보관합니다.

## 구성

| 파일 | 역할 |
| --- | --- |
| `data_analysis.ipynb` | 원본 데이터 탐색 및 CSV 생성 |
| `tmp.py` | 희귀 단어 전처리와 영어·한국어 SentencePiece 토크나이저 학습 |
| `wt.py`, `temp.py` | 토크나이저 추가 실험 및 확인 |
| `train.py` | `NHNDQ/nllb-finetuned-en2ko` 기반 fine-tuning |
| `train2.py` | 직접 구현한 Transformer를 처음부터 학습 |
| `train3.py`, `train4.py` | 기존 Transformer 체크포인트에서 추가 학습 |
| `pipeline.py` | 번역 모델 로딩과 평가 실험 |
| `utils/model.py` | PyTorch encoder-decoder Transformer 및 위치 인코딩 |
| `utils/dataset.py`, `utils/code.py` | 데이터셋, greedy decoding, BLEU·SacreBLEU·METEOR 계산 |
| `utils/paths.py` | 데이터·토크나이저·체크포인트 공통 경로 설정 |

## 데이터와 환경

데이터와 모델 가중치는 포함하지 않습니다. AI Hub 데이터는 별도로 준비하고, CSV에는 `en`과 `ko` 열이 필요합니다. 원본 JSON 폴더의 기본 위치는 `data/`이며, CSV의 기본 위치는 저장소 루트의 `en2ko.csv`입니다.

주요 Python 의존성은 PyTorch, Transformers, Accelerate, pandas, NumPy, scikit-learn, SentencePiece, Evaluate, SacreBLEU, NLTK, tqdm, W&B입니다. GPU를 사용할 경우 환경에 맞는 PyTorch를 먼저 설치하세요. 기존 Transformers API를 사용하는 실험 코드이므로 아래 설치 예시는 4.x 계열을 사용합니다.

```bash
pip install 'transformers>=4.41,<5' 'accelerate>=0.26' pandas numpy scikit-learn sentencepiece evaluate sacrebleu nltk tqdm wandb
```

이 저장소는 과거 실험 아카이브이며 전체 학습 환경을 잠근 재현 패키지는 아닙니다. 학습에 사용했던 데이터·토크나이저·체크포인트를 함께 준비해야 합니다. 이번 정리에서는 모델 학습이나 번역 성능을 다시 측정하지 않았습니다.

## 경로 설정

실행 코드는 현재 작업 디렉터리 대신 저장소 위치를 기준으로 자료를 찾습니다. 다른 경로는 환경변수로 지정합니다. 상대 경로를 지정하면 저장소 루트를 기준으로 해석합니다.

| 환경변수 | 기본값 | 용도 |
| --- | --- | --- |
| `E2K_DATA_ROOT` | `data/` | 노트북에서 읽는 원본 JSON 폴더 |
| `E2K_DATA_CSV` | `en2ko.csv` | 학습·평가 CSV |
| `E2K_TOKENIZER_ROOT` | 저장소 루트 | SentencePiece 모델과 텍스트 코퍼스 |
| `E2K_MODEL_ROOT` | `model/` | 학습 결과 저장 위치 |
| `E2K_PRETRAINED_CHECKPOINT` | `model/en2ko/checkpoint-64110` | 평가용 fine-tuning 체크포인트 폴더 |
| `E2K_CUSTOM_CHECKPOINT` | `model/self/last_model.pt` | 평가용 scratch 모델 가중치 |
| `E2K_RESUME_CHECKPOINT` | 스크립트별 기본 파일 | 추가 학습에서 읽을 가중치 |
| `E2K_CUDA` | `0` | 기본 GPU 번호 |

```bash
export E2K_DATA_CSV=/path/to/en2ko.csv
export E2K_TOKENIZER_ROOT=/path/to/tokenizers
export E2K_MODEL_ROOT=/path/to/models
```

토크나이저 이름은 실험별로 다릅니다. `train2.py`·`train4.py`·`pipeline.py`는 `corpus_en.model`과 `corpus_ko.model`, `train3.py`는 `en.model`과 `ko.model`을 사용합니다. 공통 데이터셋 모듈을 가져오는 코드에는 별도로 `en2ko.model`도 필요합니다. 체크포인트와 학습 당시의 어휘 및 모델 크기를 맞춰 사용하세요.

## 실행 예시

저장소 루트에서 실행합니다. 노트북도 이 위치에서 시작하세요.

```bash
# CSV가 준비된 뒤 코퍼스와 토크나이저 생성
python tmp.py

# 사전학습 모델 fine-tuning
E2K_CUDA=0 python train.py --seed 42

# Transformer scratch 학습 (필요한 토크나이저 준비 후)
python train2.py --cuda 0 --seed 42 --name best_model.pt

# 기존 모델을 이어서 학습
E2K_RESUME_CHECKPOINT=/path/to/final_model.pt python train3.py --cuda 0

# 실제 fine-tuning 결과와 scratch 가중치를 지정하여 평가
E2K_PRETRAINED_CHECKPOINT=/path/to/checkpoint \
E2K_CUSTOM_CHECKPOINT=/path/to/last_model.pt \
E2K_CUDA=0 python pipeline.py
```

`train.py`와 `train2.py`는 W&B 로깅을 사용합니다. 로컬 실행만 필요하면 `WANDB_MODE=disabled`를 지정할 수 있습니다. 추가 학습은 모델 가중치를 읽는 방식이며 옵티마이저·스케줄러 상태까지 복구하는 완전한 재개 기능은 아닙니다.

## 결과 관리

데이터, 토크나이저 모델, 학습 가중치, W&B 로그는 `.gitignore`로 제외합니다. 과거 README의 단일 BLEU 수치는 현재 저장소만으로 재계산할 평가 산출물이 없어 성능표로 제시하지 않습니다. 각 실험을 실행한 환경과 데이터 분할, 토크나이저, 체크포인트를 함께 기록한 뒤 결과를 비교해야 합니다.
