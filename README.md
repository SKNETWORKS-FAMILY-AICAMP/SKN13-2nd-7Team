### SKN13-2nd-7Team

# 병원 이탈 예측 프로젝트 
## Predicting Patient Disposition with Machine Learning



## 👥 팀원 소개 
### 팀명 : 도망가지마!!!! 🚑


| 이름  | 이미지                                       | 이메일               |
|:------:|:-------------------------------------------:|:--------------------:|
| 남궁건우 |<img src="profile/건우.png " width="80">      | srrd1357@gmail.com |
| 장진슬 | <img src="profile/진슬,png" width="80">        | gpendlr@gmail.com |
| 전진혁 | <img src="profile/진혁.png" width="80">       | jinhyeok2844@naver.com |
| 홍채우 | <img src="profile/채우.png" width="80">   | codn1016@gmail.com |



### WBS

| 작업 명             | 시작일 | 종료일 | 담당자                | 산출물                 | 의존 작업           |
|------------------|:------:|:------:|-------------------|----------------------|------------------|
| 프로젝트 주제 선정    | 05-12 | 05-12 | 남궁건우, 전진혁, 장진슬, 홍채우                 | 없음                  | 없음              |
| 데이터 전처리        | 04-09 | 04-09 | 남궁건우, 전진혁, 장진슬  | 코드                  | csv파일    |
| 모델링             | 04-09 | 04-09 | 남궁건우, 전진혁         | 코드                  | csv파일    |
| 이진분류 및 모델링 시각화| 04-10 | 04-10 | 장진슬, 전진혁                |  코드                | 데이터 전처리, 모델링  |
| 코드 취합           | 04-10 | 04-10 | 전진혁, 남궁건우                | DB 데이터              | 데이터 전처리, 모델링, 시각화 |
| Streamlit 구현     | 05-15 | 05-16 | 홍채우               | Streamlit 화면         | 코드 기반  |
| README.md 작성     | 05-15 | 05-16 | 장진슬                | GitHub README.md      | GitHub  |
| 최종 점검           | 05-12 | 05-16| 남궁건우, 전진혁, 장진슬, 홍채우                  | -               | -  |


---

# 1. 프로젝트 개요 
   
## 📌 주제 선정 이유
### 👀 문제 인식
병원에서는 환자가 어떤 방식으로 퇴원하는지(Patient Disposition)에 따라 향후 수익과 운영 효율성에 큰 영향을 받는다.


> 특히 다음과 같은 이탈 유형 퇴원은 병원의 경영에 부정적인 영향을 준다:

> Left Against Medical Advice (LAMA)
> → 의사의 권고를 무시하고 자의로 퇴원하는 경우
>
> 이러한 환자군은 병원 수익 손실, 병상 회전율 저하, 의료 품질 지표 악화로 이어질 수 있다.

<br/><br/>



### 💡 이탈 환자 정의
다음의 퇴원 유형을 **이탈 환자 (class 1)**로 정의했다:

| 환자 유형                             | 사유             |
| --------------------------------- | -------------- |
| Left Against Medical Advice       | 의사 권고 없이 자의 퇴원 |


이외의 퇴원(단기 병원 전원, 재환 시설 전원, 사망 등)은 **정상 퇴원 (class 0)**으로 분류했다.

<br/><br/>

---

### 🔍 프로젝트 소개, 목표 
🔗 목표: 이탈 환자 예측하는 이진 분류 모델 개발

📂 사용 데이터: 뉴욕주 병원 입원 퇴원 데이터 (2010년)\
↳ Kaggle dataset: hospital-inpatient-discharges-sparcs.csv

<br/><br/>

### 논문기반으로 수익성, 얘기 ## 🧭 프로젝트 필요성

---

# 2. 프로젝트 구 조 ~~!@  -진혁 
# 2. 프로젝트 

## 2-1. 사용한 기술 스택
<br/><br/>
<p align="center">
  <!-- 버전 관리 -->
  <img src="https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white"/>
  <img src="https://img.shields.io/badge/github-181717?style=for-the-badge&logo=github&logoColor=white"/>

  <!-- 언어 및 환경 -->
  <img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white"/>
  <img src="https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
  <img src="https://img.shields.io/badge/numpy-013243?style=for-the-badge&logo=numpy&logoColor=white"/>
  <img src="https://img.shields.io/badge/matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white"/>
  <img src="https://img.shields.io/badge/seaborn-0F1111?style=for-the-badge&logo=seaborn&logoColor=white"/>

  <!-- 머신러닝 라이브러리 -->
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white"/>
  <img src="https://img.shields.io/badge/XGBoost-FF6600?style=for-the-badge&logo=xgboost&logoColor=white"/>
  <img src="https://img.shields.io/badge/imbalanced--learn-FF6F00?style=for-the-badge&logo=scikitlearn&logoColor=white"/>

  <!-- 전처리 관련 -->
  <img src="https://img.shields.io/badge/preprocessing-blueviolet?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/SMOTE-red?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/OrdinalEncoder-yellow?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/OneHotEncoder-green?style=for-the-badge"/>

  <!-- 모델 -->
  <img src="https://img.shields.io/badge/LogisticRegression-9B59B6?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/RandomForest-27AE60?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/XGBoost-E74C3C?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/DecisionTree-F1C40F?style=for-the-badge"/>
</p>
<br/><br/>



# 3. 🧼 데이터 전처리
## 3-1. 비정상 데이터 변환

- Length of Stay → 문자열 제거 후 숫자 변환

- NaN이 많은 열 (10,000건 이상 결측) 제거

## 3-2. 의미 없는 컬럼 제거

- Zip Code, License Number, Certificate Number 등 식별 불가능하거나 분석과 무관한 컬럼 제외

## 3-3. 범주형/수치형 변수 분류

- 수치형: Total Charges, Total Costs, Length of Stay

- 범주형: 입원 경로, 진단 코드, 성별 등

## 3-4. 이상치 제거 & 희귀 범주 통합

- 정규분포 Z-score 기반 이상치 제거

- 빈도 낮은 범주는 others 통합

<br/><br/>

---

# 4. 🤖 모델링 및 학습 전략
## 4-1. 전처리 
 ### 4-1-1. 범주형 변수 처리
- Tree 기반 모델: OrdinalEncoder 사용 (순서 인코딩)
- 선형 모델 (Logistic Regression): OneHotEncoder 사용 (가변수 처리)

 ### 4-1-2. 수치형 변수 처리

 ### 4-1-3. KNN을 통한 결측치 대체 (KNNImputer)

 ### 4-1-4. 정규화 (StandardScaler)

 ### 4-1-5. 클래스 불균형 처리
소수 클래스인 이탈 환자를 더 잘 학습하도록 SMOTE 기법을 적용하여 과소표집 없이 오버샘플링

<br/><br/>

## 4-2. 모델링과 학습 




<br/><br/>



| 지표        | 설명                               |
| --------- | -------------------------------- |
| Accuracy  | 전체 예측 정확도                        |
| Precision | 이탈이라고 예측한 것 중 실제 이탈의 비율          |
| Recall    | 실제 이탈 중 예측에 성공한 비율 (이탈 탐지율)      |
| F1 Score  | Precision과 Recall의 조화 평균 (균형 지표) |

<br/><br/>


📊 성능 요약

| Model               | Accuracy | Precision | Recall | F1 Score |
| ------------------- | -------- | --------- | ------ | -------- |
| Logistic Regression | 0.9790   | 0.5357    | 0.5357 | 0.5357   |
| Random Forest       | 0.9839   | 0.9706    | 0.2946 | 0.4521   |
| XGBoost             | 0.9833   | 0.8085    | 0.3393 | 0.4780   |



> ### Logistic Regression: 
> recall과 precision이 균형 잡힘. 단순하고 해석 용이함.

> ### Random Forest
> 정밀도는 높지만 recall이 낮음. 실제 이탈을 놓칠 가능성 존재.

> ### XGBoost
> recall, precision의 균형 측면에서 우수. F1 Score 최고.

>> F1 Score 기준으로 가장 균형 잡힌 성능을 보인 **XGBoost**가 실용성과 성능 모두에서 우수하다고 인식.


<br/><br/>



💡 주요 해석 및 인사이트
이탈 환자는 소수 클래스이기 때문에 모델이 예측하기 어려운 문제.

Precision이 높아도 Recall이 낮으면 실제 이탈 환자를 놓칠 수 있으므로, 
의료 안전과 예방 목적이라면 Recall을 높이는 방향의 threshold 조정이 중요하다. 
(병원 입장에서는 실제 이탈 환자를 최대한 잡아내는 것이 중요하기 때문에, recall이 낮은 모델은 주의해야함)

SHAP 해석을 통해 어떤 변수들이 이탈 예측에 영향을 주는지도 확인

**Length of Stay**, **Type of Admission**, **Severity of Illness** 등이 주요 변수로 작용함을 확인??? 이거 다시 확인 





---
# 💭 한 줄 회고

| 이름 | 한 줄 회고 |
|:------:|-------------------|
| 건우 | |
| 진슬 | |
| 진혁 | |
| 채우 | |




<br/><br/>
<br/><br/>
<br/><br/>


끝.










