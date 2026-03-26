### 뇌졸중 후 마비말장애 평가를 보조하기 위해 정상 발화 기반 한국어 STT baseline을 구축하고,<br>자소 단위 전사를 통해 발음 오류를 정량화할 가능성과 실제 환경 일반화 한계를 함께 분석한 프로젝트입니다.

자소 단위 전사를 활용해 발음 오류를 정량적으로 확인할 수 있는지 탐색했으며, 동시에 실제 녹음 환경에서의 일반화 한계도 함께 분석했습니다.<br>
내부 Test set 기준 각 모델별 평균 CER은 다음과 같았습니다.<br>
- DeepSpeech2: 0.3652
- Simple-Attention: 0.267403
- Transformer: 0.2603

아래 사이트에 접속하시면 Simple-Attention모델과 Transformer모델을 사용해 보실 수 있습니다.<br>
https://gmrmsy.github.io/STT_test_site_1/


# 개요

본 프로젝트는 뇌졸중 등 뇌손상 이후 나타나는 언어장애 평가를 보조하기 위한 음성 기반 AI 시스템을 목표로 시작했습니다.
제가 맡은 영역은 마비말장애 검사와 연관된 발성·조음 능력 평가였으며 발화를 텍스트로 안정적으로 전사할 수 있다면 정상 발화 대비 발음 오류를 정량적으로 관찰할 수 있다고 판단했습니다.
다만 STT 결과를 임상 점수와 직접 동일시하기보다는 정상 발화 대비 오류 정도를 측정하기 위한 1차 보조지표로 활용하는 방향이 더 적절하다고 보았습니다.
특히 이 프로젝트에서는 음절 단위보다 더 세밀한 발음 차이를 반영하기 위해 자소 단위 전사(ex. 안녕 → ㅇㅏㄴㄴㅕㅇ)를 적용했습니다.<br>

<br>

<img width="500" height="223" alt="Image" src="https://github.com/user-attachments/assets/b83eff57-bacd-4fa8-ad02-4f8407515197" />

<br>

때문에 언어 평가 모델 개발 프로젝트를 진행하며 동시에 STT를 구축하기 위해 병행하며 진행했습니다.
영어 기반의 STT는 흔히 있지만 한글 기반의 Pretrain된 모델은 Kospeech외에 찾기가 어려웠습니다.
때문에 DeepSpeech2 기반의 Kospeech모델을 토대로 STT 모델을 구축하기 시작했습니다.

해당 STT모델 개발하기 위해 제가 수행한 역할은 다음과 같습니다.
1) AI Hub 음성 데이터셋 조건 정의 및 학습 데이터 선별
2) 한국어 자소 단위 토큰화 및 전처리 파이프라인 구성
3) DeepSpeech2 기반 baseline 구현
4) GRU + Self-Attention 기반 Simple-Attention 모델 구현
5) Transformer 기반 STT 모델 구현 및 성능 비교
6) CER(Character Error Rate) 중심 정량 평가 및 오류 분포 분석
  
<br>

## 데이터 선정 및 전처리
본 프로젝트의 이름에서 알 수 있듯 뇌졸중 등 뇌손상으로 인한 환자들의 검사를 진행하기 때문에 대부분의 대상자들이 60대 이상의 노인분들입니다.
때문에 STT 모델 학습에 사용될 데이터도 AIhub에서 제공하는 '자유대화 음성(노인남여)'를 사용하였습니다.
해당 데이터는 1,000명 이상의 발화자를 대상으로 3,000여 시간 이상의 음성데이터로 이루어져있습니다.
전체 데이터는 약 306.87GB 규모였기 때문에 Colab 환경에서 전체 학습은 현실적으로 어려웠고 아래 조건으로 데이터를 정제했습니다.

1) 검사 항목에 사용하는 발음을 포함하는 문장
2) 검사 항목과 길이가 비슷한 문장 (2초 초과, 10초 이하)
3) 불명확 발화 특수 토큰((SP:대상포), (NO:첨단지구), (FP:뭐))이 포함되지 않은 문장

<a href="https://github.com/gmrmsy/STT/blob/main/1)data_select.ipynb">데이터 선정 코드 보기</a>

이 기준으로 총 141,658개 발화를 선별했습니다.
또한 동일 문장이 여러 화자에게 반복 녹음된 구조를 고려해 문장 기준으로 Train / Validation / Test = 8 : 1 : 1 분할을 적용했습니다.

<a href="https://github.com/gmrmsy/STT/blob/main/2)data_preprocessing.ipynb">데이터 전처리 코드 보기</a>

## 모델 구성

### 1. DeepSpeech2
<img width="500" height="490" alt="Image" src="https://github.com/user-attachments/assets/b7c46ed7-8909-4ebc-bd3e-65bb1030215b" />

CNN으로 Mel-Spectrogram 특징을 추출하고, GRU 기반 RNN으로 시계열 정보를 학습한 뒤, CTC Loss를 통해 입력-출력 정렬 없이 문장을 예측하는 baseline 모델입니다.<br>
<a href="https://github.com/gmrmsy/STT/blob/main/3)ds2_train.ipynb">DeepSpeech2 모델 구현 및 학습 코드 보기</a>

### 2. Simple-Attention
<img width="500" height="356" alt="Image" src="https://github.com/user-attachments/assets/6825b908-8e49-42d0-8223-3db24fae4d51" />

DeepSpeech2의 CNN + GRU 흐름을 유지하되, Self-Attention을 추가하여 자소/문자 단위 예측에 유리한 시퀀스 표현을 학습하도록 설계했습니다.<br>
<a href="https://github.com/gmrmsy/STT/blob/main/4)Simple_Attention_train.ipynb">Simple-Attention 모델 구현 및 학습 코드 보기</a>

### 3. Transformer
<img width="500" height="721" alt="Image" src="https://github.com/user-attachments/assets/89d68095-e49a-4a04-9020-48bb3d5e95de" />

Attention 기반 Encoder–Decoder 구조를 STT에 적용한 모델입니다.<br>
입력은 음성 특징, 출력은 자소 토큰 시퀀스로 구성하여, 장거리 의존성과 문맥 반영 능력을 강화하고자 했습니다.<br>
<a href="https://github.com/gmrmsy/STT/blob/main/5)Transformer_train.ipynb">Transformer 모델 구현 및 학습 코드 보기</a>

## 주요결과

<DeepSpeech2, Simple-Attention, Transformer 모델의 CER 분포><br>
<img width="500" height="377" alt="Image" src="https://github.com/user-attachments/assets/daf68b57-f6a4-46dc-aea4-887d3cf384e6" />

내부 Test set 기준 평균 CER은 다음과 같았습니다.
- DeepSpeech2: 0.3652
- Simple-Attention: 0.267403
- Transformer: 0.2603

Transformer가 평균 및 중앙값 기준으로 가장 낮은 CER을 보였으며, CER 0 샘플 수 역시 3,459개로 가장 많았습니다.
반면 분산은 Transformer가 가장 크게 나타나, 일부 샘플에서는 성능이 급격히 무너지는 불안정성도 함께 확인되었습니다.


## 결과해석

이 실험을 통해 단순한 RNN 기반 구조보다 Attention 기반 구조가 한국어 자소 단위 STT에서 더 나은 성능을 낼 수 있음을 확인했습니다.
특히 Transformer는 내부 Test set에서 가장 우수한 평균 성능을 보였기 때문에, 자소 단위 발음 분석의 baseline 모델로서 가능성을 보여주었습니다.
하지만 실제로 직접 녹음한 외부 음성을 입력했을 때는 내부 Test set보다 전사 품질이 크게 저하되었습니다.
이는 데이터의 다양성(Variety)의 부족으로 새로운 화자 및 실제 녹음 환경에 대한 일반화 성능의 한계로 해석할 수 있습니다.


## 한계분석

본 프로젝트는 문장 기준으로 데이터를 8:1:1로 분할했습니다. 이 방식은 데이터셋 내부 문장 커버리지를 확보하는 데는 유리하지만 학습에 사용되지 않은 새로운 화자에게도 안정적으로 성능을 내는 화자 독립 일반화 성능을 충분히 검증하기에는 한계가 있습니다. 또한 발화자들의 성향에 따라 자연스러운 발화와 책을 읽는 듯한 딱딱한 발화가 섞여있어 이 두 발화의 특성을 구분하기에 다양성이 부족하다는 한계가 있습니다.


## 향후 개선 방향

향후에는 직접 구현한 모델의 개선과 별개로 pretrained된 오픈소스 STT 모델을 활용하는 또 다른 방법도 검토하고자 합니다.
이를 통해 보다 안정적인 전사 결과를 확보하고 뇌졸중 후 마비말장애 평가 보조에 적합한 방향을 비교·탐색할 수 있을 것으로 기대합니다.
