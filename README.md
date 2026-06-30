# 🌐 E2K_Translation: A Glorious Failure

> "거인의 어깨 위에 올라타지 않으면 얼마나 바닥을 기게 되는가"에 대한 장엄한 실험적 증명.
> 영어-한국어 번역 모델을 **사전 학습된 가중치(Pre-trained Checkpoints) 없이 스크래치(From-scratch)로 학습**시켰을 때 벌어지는 대참사를 기록한 레포지토리입니다.

## 📝 TL;DR
- **Task:** English to Korean Translation
- **Approach:** No Checkpoints, Pure From-Scratch Training
- **Result:** BLEU Score 1.x (번역기가 아니라 외계어 생성기를 만들었습니다.)
- **Lesson Learned:** Transfer Learning(전이 학습)은 인류의 위대한 발명품입니다. Hugging Face에 감사합시다.

## 🚀 Model & Training
이 프로젝트의 가장 큰 특징은 **'근거 없는 패기'**였습니다.
최신 번역 모델 아키텍처를 가져왔지만, 수많은 데이터로 정제된 훌륭한 Pre-trained 가중치를 과감히(?) 버리고 백지상태에서 학습을 진행했습니다. 

- **Architecture:** [여기에 모델 이름 작성, ex: Transformer / Seq2Seq]
- **Dataset:** [사용했던 데이터셋 대충 작성, ex: AI Hub 영한 번역 데이터 일부]
- **Initialization:** Random Initialization (From-scratch)

## 📊 Results & Analysis (Post-Mortem)

### Metric
* **BLEU Score:** `1.x` 

### Qualitative Analysis
영어를 입력하면 한국어 단어들이 아무런 맥락 없이 무작위로 튀어나오는 신비로운 현상을 관찰할 수 있습니다. 문법과 의미는 완전히 붕괴되었으며, 자연어 처리(NLP)가 아니라 '자연어 파괴'에 가깝습니다.

### 🔍 Why did it fail? (실패 원인 분석)
1. **사전 학습(Pre-training)의 절대적 부재:** 번역과 같은 복잡한 언어 이해 태스크에서 스크래치 학습은 갓난아기에게 셰익스피어 희곡을 쓰라고 강요하는 것과 같습니다. 언어의 기본적인 통계적 구조를 전혀 모르는 상태에서 매핑만 시도했기 때문에 성능이 나올 수 없었습니다.
2. **데이터와 컴퓨팅 파워의 한계:** 스크래치로 유의미한 성능을 내려면 천문학적인 규모의 코퍼스와 막대한 GPU 리소스가 필요하지만, 당시의 소박한 환경은 이를 뒷받침하지 못했습니다.

## 💡 Conclusion
이 레포지토리는 비록 번역기로서의 실용성은 0에 수렴하지만, 딥러닝에서 **Weight Initialization과 Pre-trained Model이 얼마나 압도적으로 중요한지**를 온몸으로 증명해 낸 훌륭한 오답 노트입니다. 

---
*과거의 화려한 삽질은 미래의 훌륭한 논문 실적을 위한 거름이 됩니다.*


# AI Hub Korean-English Translation Task

이 저장소는 과거에 **AI 허브(AI Hub)**의 한영 번역 데이터를 활용하여 번역 모델 학습 및 파이프라인 구축을 테스트했던 프로젝트 아카이브입니다. 

## 프로젝트 개요
- **데이터셋**: AI Hub 한영/영한 번역 데이터
- **주요 내용**: 데이터 전처리, 번역 모델 파이프라인 구성 및 학습 테스트
- **비고**: 과거 작업 기록 보존(아카이빙)을 위해 업로드된 프로젝트입니다. 용량이 큰 데이터셋(`*.csv`, `*.txt`)과 모델 가중치(`*.model`), `wandb` 로그 등은 `.gitignore`를 통해 제외되었습니다.
