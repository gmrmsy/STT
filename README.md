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
<img width="500" height="26" alt="Image" src="https://github.com/user-attachments/assets/2a22d4b9-bc82-4d30-a4c8-8ed0a110272e" />

<예측 문장><br>
<img width="500" height="24" alt="Image" src="https://github.com/user-attachments/assets/98536fd2-4325-4665-9c78-c3bf6998047a" />

위 처럼 DeepSpeech2를 기반으로 Tensorflow로 모델을 구축할 경우 파라미터값이 매우 많은걸 볼 수 있습니다.
물론 메이저한 딥러닝 모델들과 비교한다면 굉장히 적은 편이지만 리소스와 컴퓨팅 파워가 부족한 입장에서 이 정도의 파라미터도 학습에 부담이 됩니다.
또한 학습을 진행했을 때 예측 문장의 수준 또한 고려 대상이 되지 않을 정도로 좋지 않은것을 볼 수 있습니다.
때문에 이 구조에서 더 개선된 모델을 구축하기 위해 단순히 RNN레이어의 반복을 하지 않고 RNN레이어와 Attention 레이어 함께 사용하여 성능을 높이고 파라미터를 줄여보았습니다.

<br>

<img width="500" height="940" alt="Image" src="https://github.com/user-attachments/assets/0eff62db-cd50-44c5-8429-af1647db028a" />

위 이미지에서 보이듯 파라미터값이 크게 줄어들어 학습에 부담이 없게 되었습니다.<br>

## 데이터 선정
본 프로젝트의 이름에서 알 수 있듯 뇌졸중 등 뇌손상으로 인한 환자들의 검사를 진행하기 때문에 대부분의 대상자들이 60대 이상의 노인분들입니다.
때문에 STT 모델 학습에 사용될 데이터도 AIhub에서 제공하는 '자유대화 음성(노인남여)'를 사용하였습니다.

## 1차 모델 구축
꾸준히 데이터 정제, 선별과 모델 구조 조정을 했지만 프로젝트 기간내에 학습을 완료한 모델의 예측결과는 기대에 미치지 못했습니다.
전사 문장과 예측 문장 사례는 아래와 같습니다.

<전사 문장><br>
<img width="500" height="26" alt="Image" src="https://github.com/user-attachments/assets/2a22d4b9-bc82-4d30-a4c8-8ed0a110272e" />

<예측 문장><br>
<img width="500" height="23" alt="Image" src="https://github.com/user-attachments/assets/614ed6fe-bb56-4e00-981d-d98b8af85a10" />

이전 RNN모델과 비교한다면 개선된 모습이 모이지만 모델이 음성을 문자로 온전히 전사시켜야지만 모델의 결과를 통해 장애정도를 판단할 수 있습니다.
하지만 현재 전사된 문장도 정상발음을 제대로 전사하지 못하는 모습을 볼 수 있습니다.
때문에 프로젝트 종료 후 Transformer를 학습하여 모델에 적용시킬 예정입니다.

자세한 코드 전문 파일은 레포지토리에 저장되어 있습니다.
