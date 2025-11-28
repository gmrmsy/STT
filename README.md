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
<img width="500" height="24" alt="Image" src="https://github.com/user-attachments/assets/234ad4ed-010d-499c-9370-fa855af7e684" />
<img width="500" height="40" alt="Image" src="https://github.com/user-attachments/assets/aa34fee3-420d-4e01-82fd-d444d66b748b" />

<예측 문장><br>
<img width="500" height="23" alt="Image" src="https://github.com/user-attachments/assets/6b7af473-c2fe-4f0c-a0d4-7a87610e76d4" />
<img width="500" height="26" alt="Image" src="https://github.com/user-attachments/assets/7b5fb8bf-5ec1-4956-8aae-4d9a774294b2" />

위 처럼 DeepSpeech2를 기반으로 Tensorflow로 모델을 구축할 경우 파라미터값이 매우 많은걸 볼 수 있습니다.
물론 메이저한 딥러닝 모델들과 비교한다면 굉장히 적은 편이지만 리소스와 컴퓨팅 파워가 부족한 입장에서 이 정도의 파라미터도 학습에 부담이 됩니다.
학습을 진행했을 때 예측 문장이 어느정도 전사가 잘 되지만 전사된 문장을 기준으로 점수화를 하기에는 더 정교한 정확도가 필요해보입니다.
때문에 이 구조에서 더 개선된 모델을 구축하기 위해 단순히 RNN레이어의 반복이 아닌 Transformer를 학습하여 모델에 적용시켜 성능을 높여볼 예정입니다.
