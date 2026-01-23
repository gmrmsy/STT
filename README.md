# 개요

뇌졸중 후 언어장애 진단을 위한 딥러닝 기반 언어 기능 평가 서비스 개발(이하 언어 평가 모델 개발) 프로젝트 진행 중 제가 맡은 검사항목들은 마비말장애 검사에 해당하는 항목들이었습니다. <br>
인지-언어 연결 능력(듣기·이해·표현)을 평가하는 실어증 항목과 달리 마비말장애 검사 항목은 발성·조음 및 구강 근육 사용 능력을 평가하는 항목이기 때문에 음성을 통한 정확한 텍스트 전달이 중점된 항목이라 생각했습니다.
때문에 STT모델을 통해 음성을 텍스트로 전환하고 그 전환된 텍스트의 정확도가 곧 검사점수와 같아야한다고 판단했습니다.
더 나아가 자음과 모음의 디테일한 발음을 확인하는 검사인 만큼 음절, 단어, 형태소 등으로 전사된 STT보다 자소단위(ex. 안녕 -> ㅇㅏㄴㄴㅕㅇ)로 전사된 STT를 사용하는 것이 적합하다고 판단했습니다.

<img width="500" height="223" alt="Image" src="https://github.com/user-attachments/assets/b83eff57-bacd-4fa8-ad02-4f8407515197" />

<br><br>

때문에 언어 평가 모델 개발 프로젝트를 진행하며 동시에 STT를 구축하기 위해 병행하며 진행했습니다.
영어 기반의 STT는 흔히 있지만 한글 기반의 Pretrain된 모델은 Kospeech외에 찾기가 어려웠습니다.
때문에 DeepSpeech2 기반의 Kospeech모델을 토대로 STT 모델을 구축하기 시작했습니다.

## 데이터 선정 및 전처리
본 프로젝트의 이름에서 알 수 있듯 뇌졸중 등 뇌손상으로 인한 환자들의 검사를 진행하기 때문에 대부분의 대상자들이 60대 이상의 노인분들입니다.
때문에 STT 모델 학습에 사용될 데이터도 AIhub에서 제공하는 '자유대화 음성(노인남여)'를 사용하였습니다.
해당 데이터는 1,000명 이상의 발화자를 대상으로 3,000여 시간 이상의 음성데이터로 이루어져있습니다.
하지만 총 306.87 GB의 모든 데이터를 사용하는것은 컴퓨터 리소스적인 부분에서 불가능하다 판단했습니다.
때문에 몇 가지 조건을 가지고 데이터를 정제하여 사용하였습니다.

1) 검사 항목에 사용하는 발음을 포함하는 문장
2) 검사 항목과 길이가 비슷한 문장 (2초 이상 AND 10초 이하)
3) 전사된 문장에 (SP:대상포), (NO:첨단지구), (FP:뭐) 등 불명확한 음성에 대한 특수토큰을 포함하지 않는 문장

이렇게 세 가지 조건에 해당하는 문장을 선별하여 총 141,658개의 데이터를 가지고 학습을 진행하였습니다.

## DeepSpeech2
<img width="500" height="490" alt="Image" src="https://github.com/user-attachments/assets/b7c46ed7-8909-4ebc-bd3e-65bb1030215b" />

DeepSpeech2의 구조는 위 그림과 같습니다.
Convolutaion Layer를 통해 Mel-Spectrogram의 이미지적 특징을 추출합니다. 이는 음성적 특징을 추출하는 것과 흡사하다 할 수 있습니다.
특징이 추출된 데이터는 시간의 흐름에 따라 바뀌는 발화를 분석하기 위해 Recurrent Neural Network(GRU)를 거치게 됩니다.
RNN 레이어를 거친 데이터를 Fully Conected Layer로 처리합니다.
또한 STT 모델의 경우 각각의 스텝에 음성과 텍스트가 정렬되어서 나오지 않기 때문에 Loss값을 계산할 때 CTC Loss Function을 사용합니다. 

<br>

<img width="500" height="52" alt="Image" src="https://github.com/user-attachments/assets/282f2db8-1e86-4d0c-99dd-c2bbfe809118" />
<br>
<전사 문장><br>
<img width="456" height="15" alt="Image" src="https://github.com/user-attachments/assets/8a7f1dec-1d4c-4f59-8bbb-00b98a1326f6" />

<예측 문장><br>
<img width="462" height="15" alt="Image" src="https://github.com/user-attachments/assets/1dd12a47-3fb9-4060-8953-ce5a45f5290b" />

<DeepSpeech2 모델의 CER 분포><br>
<img width="500" height="377" alt="Image" src="https://github.com/user-attachments/assets/ac17cca9-0a48-4788-acec-6cb07c4a7a10" />

위 처럼 DeepSpeech2를 기반으로 Tensorflow로 모델을 구축할 경우 파라미터값이 매우 많은걸 볼 수 있습니다.
물론 메이저한 딥러닝 모델들과 비교한다면 굉장히 적은 편이지만 리소스와 컴퓨팅 파워가 부족한 입장에서 이 정도의 파라미터도 학습에 부담이 됩니다.
학습을 진행했을 때 예측 문장이 어느정도 전사가 잘 되지만 DeepSpeech2의 평균 CER을 확인한 결과 0.3652으로 확인되어 좀 더 정교한 정확도가 필요해보입니다.
때문에 이 구조에서 더 개선된 모델을 구축하기 위해 단순히 RNN레이어의 반복이 아닌 Transformer의 어텐션 기술사용해 모델을 구축 후 학습하여 성능을 높여볼 예정입니다.

## Simple-Attention
<img width="500" height="356" alt="Image" src="https://github.com/user-attachments/assets/6825b908-8e49-42d0-8223-3db24fae4d51" />

위 이미지는 Convolution Layer, Recurrent Neural Network(GRU), Self_Attention을 사용해 구축한 STT모델을 도식화한 이미지입니다.
DeepSpeech2의 Convolution Layer를 통해 Mel_spectrogram의 이미지적 특징을 추출 후 GRU Layer로 이어지는 흐름을 차용하여 그 후 Self-Attention과 GRU Layer를 반복해 모델을 구축했습니다.
이로인해 음소/문자 단위 예측에 유리한 시퀀스 표현(representation)을 학습하고, 인식 정확도를 높이기 위한 특징을 추출하였습니다.

<br>

<img width="500" height="56" alt="Image" src="https://github.com/user-attachments/assets/6c915ca9-fa53-4f75-a076-b0d33db28df2" />
<br>
<전사 문장><br>
<img width="456" height="15" alt="Image" src="https://github.com/user-attachments/assets/8a7f1dec-1d4c-4f59-8bbb-00b98a1326f6" />

<예측 문장><br>
<img width="463" height="15" alt="Image" src="https://github.com/user-attachments/assets/bd5bda8b-dfb7-42bb-a42b-363f36a21e15" />

<DeepSpeech2, Simple-Attention 모델의 CER 분포><br>
<img width="500" height="377" alt="Image" src="https://github.com/user-attachments/assets/33f8158d-a8bb-42a5-82e3-33bcf7e75456" />

위 그래프에서 알 수 있듯 DeepSpeech2 모델보다 Attention을 사용한 모델의 CER이 더 낮게 분포되어있는것을 알 수 있습니다.
실제 수치를 봤을 때 Simple-Attention 모델의 평균 CER은 0.267403으로 DeepSpeech2 모델보다 더 성능이 향상되었고 추후 발전의 가능성을 볼 수 있다고 생각이 들었습니다.
하지만 프로젝트의 기간이 다 되어 프로젝트 중 진행한 STT모델의 개발은 여기서 멈춰졌고 결국 프로젝트에는 사용할 수 없었습니다.
때문에 프로젝트 후 계속해서 Transformer와 다른 STT 모델의 특성을 조사하여 발전시켜나아갈 예정입니다.


## Transformer
<img width="500" height="721" alt="Image" src="https://github.com/user-attachments/assets/89d68095-e49a-4a04-9020-48bb3d5e95de" />

AI를 공부한다면 모를 수 없는 Tranformer 모델의 구조입니다. 프로젝트 종료 후 Transformer에 대한 조사를 진행한 후 그 구조를 STT 모델에 적용하여 개발을 진행하였습니다.
Attention is all you need에서 사용한 모델의 구조를 똑같이 차용하였고 거기에서 Input Data에는 음성데이터, Output Data에는 전사된 문장데이터를 자소토큰으로 토큰화한 데이터를 넣어서 학습을 진행했습니다.

<전사 문장><br>
<img width="456" height="15" alt="Image" src="https://github.com/user-attachments/assets/8a7f1dec-1d4c-4f59-8bbb-00b98a1326f6" />

<예측 문장><br>
<img width="456" height="15" alt="Image" src="https://github.com/user-attachments/assets/1aaa09ef-c887-4658-8046-7395706f3c94" />

<DeepSpeech2, Simple-Attention, Transformer 모델의 CER 분포><br>
<img width="500" height="377" alt="Image" src="https://github.com/user-attachments/assets/daf68b57-f6a4-46dc-aea4-887d3cf384e6" />

CER을 확인해 본 결과 중앙값, 평균 등 대표값은 Transformer 모델이 제일 낮게 확인되지만 분산값은 Transformer 모델이 제일 큰 것을 확인 할 수 있습니다.
이런 결과의 원인으로는 Transformer 모델의 복잡성을 살려내기에 사용한 141,658개의 데이터(약 20GB)의 규모(Volume)가 충분하지 않았던 것으로 보입니다.
그리고 Attention 기반 Encoder–Decoder E2E STT는 입력–출력 정렬을 모델이 잠재적으로 학습해야 하므로, 단조 정렬 가정을 두고 모든 정렬을 합산하는 CTC 기반 모델보다 초기 학습이 불안정해질 수 있습니다.
또한 Transformer의 특성상 파라미터의 민감도가 높기 때문에 이에 맞는 튜닝을 다양하게 진행해야하는 것으로 보입니다.
