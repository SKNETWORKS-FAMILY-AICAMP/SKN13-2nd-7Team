### SKN13-2nd-7Team

# 환자 이탈 예측 프로젝트 
## Predicting Patient Disposition with Machine Learning
Left Against Medical Advice (LAMA) - 의학적 권고에도 불구하고 스스로 병원을 떠나는 환자


## 👥 팀원 소개 
### 팀명 : 도망가지마!!!! 🚑


| 이름  | 이미지                                       | 이메일               |
|:------:|:-------------------------------------------:|:--------------------:|
| 남궁건우 |<img src="readme_img/건우.png " width="80">      | srrd1357@gmail.com |
| 장진슬 | <img src="readme_img/진슬.png" width="80">        | gpendlr@gmail.com |
| 전진혁 | <img src="readme_img/진혁.png" width="80">       | jinhyeok2844@naver.com |
| 홍채우 | <img src="readme_img/채우.png" width="80">   | codn1016@gmail.com |



### WBS

| 작업 명             | 담당자                | 산출물                 | 의존 작업           |
|------------------|-------------------|----------------------|------------------|
| 프로젝트 주제 선정   | 장진슬, 전진혁 | 없음                  | 없음              |
| 데이터 전처리       | 남궁건우, 전진혁, 장진슬 | 코드                  | csv파일    |
| 모델링            | 남궁건우, 전진혁         | 코드                  | csv파일    |
| 이진분류 및 모델링 시각화| 장진슬, 전진혁                |  코드                | 데이터 전처리, 모델링  |
| 코드 취합           | 전진혁, 남궁건우                | DB 데이터              | 데이터 전처리, 모델링, 시각화 |
| Streamlit 구현    | 홍채우               | Streamlit 화면         | 코드 기반  |
| README.md 작성     | 장진슬                | GitHub README.md      | GitHub  |
| 최종 점검          | 남궁건우, 전진혁, 장진슬, 홍채우                  | -               | -  |


---

# 1. 프로젝트 개요 
   
## 1-1. 📌 주제 선정 이유
### 👀 문제 인식
병원에서는 환자가 어떤 방식으로 퇴원하는지(Patient Disposition)에 따라 향후 수익과 운영 효율성에 큰 영향을 받는다.


> 특히 다음과 같은 이탈 유형 퇴원은 **병원 경영에 직접적인 손실**을 준다:
>> Left Against Medical Advice (LAMA): 의학적 권고 거부 퇴원

> LAMA: 의학적 권고에도 불구하고 스스로 병원을 떠나는 환자
>>의사의 권고를 무시하고 자의로 퇴원하는 사례
>> 즉, 의료진이 입원 또는 치료를 권유했음에도 불구하고 환자 혹은 보호자가 자의적으로 퇴원을 결정하는 경우
<br/><br/>
>> LAMA 환자들은 재입원율, 합병증 발생률, 사망률 등이 높은 것으로 보고되며, 
>> 의료 기관의 수익 손실, 자원 낭비, 법적 책임 문제를 유발할 수 있어 의료 현장에서 중요한 이슈다.

<br/><br/>

## 1-2. 발생 현황 및 중요성

* 일부 병원에서는 LAMA 비율이 1~2% 수준이지만,
특정 환자군(예: 정신질환자, 알코올 중독자, 저소득층)에서는 20% 이상으로 보고되기도 한다.
* 의료 시스템 전반에 미치는 영향에도 불구하고, LAMA에 대한 연구나 대응 체계는 매우 부족한 실정.

<br/><br/>



### 💡 이탈 환자 정의 및 타겟 설정
다음의 퇴원 유형을 **이탈 환자 (class 1)** 로 정의했다:

| 구분     | 퇴원 사유                       | 클래스          |
| ------ | --------------------------- | ------------ |
| 이탈 환자  | Left Against Medical Advice | 1 (Positive) |
| 정상 퇴원자 | 일반 퇴원, 전원, 사망 등             | 0 (Negative) |


이외의 퇴원(단기 병원 전원, 재활 시설 전원, 사망 등)은 **정상 퇴원 (class 0)**으로 분류했다.
> 본 프로젝트는 이진 분류 문제로 설정하여 LAMA 여부를 예측


<br/><br/>

 
---



### 🔍 프로젝트 소개, 목표 


🔗 목표: 머신러닝 모델 향후 활용 가능성 
* **이탈 환자(LAMA) 여부**를 사전에 예측하는 머신러닝 모델 개발
* 의료기관에서는 해당 머신러닝 모델을 활용한 환자관리, 실용적인 의료 예측 시스템 구축 효과 기대
<br/><br/>

📂 사용 데이터: 뉴욕주 병원 입원 퇴원 데이터 (2010년)\
↳ Kaggle dataset: hospital-inpatient-discharges-sparcs.csv\
↳ 크기: 156만 개\
↳ 특성(컬럼): 총 37개 (범주형/수치형 혼합) / 이후 사용한 컬럼은 29개

<br/><br/>


---

# 2. 프로젝트 구조
```
SKN13-2nd-7Team
├── raw_data/
│   └── hospital-inpatient-discharges-sparcs-de-identified-2010-1.csv
│
├── models/
│   ├── LogisticRegression_model.pkl              # 로지스틱 회귀 모델 (F1: )
│   ├── RandomForestClassifier_model.pkl        # 랜덤 포레스트 분류 모델 (F1: )
│   ├── XGBClassifier_model.pkl             	    # XGBoost 분류 모델 (F1: )
│   └── GradientBoosting_model.pkl
│
├── preprocessor/
│   ├── preprocessor.pkl
│   └── preprocessor2.pkl                     
| 
├── utils/
│   ├── cate_outlier.py
│   ├── num_outlier.py
│   ├── get_best_model.py
│   ├── preprocess_drop.py
│   ├── preprocessing_utils.py
│   └── model_trainer.py
│
├── images/
|    ├── PR_Curve.png
|    ├── Rader_Chart.png
|    ├── Num_columns_Outliers.png
│    └── ROC_Curve.png
│
│
├── 산출물/
│   ├── XGBoost - Confusion Matrix.png
│   ├── XGBoost - Feature Importance (Top 10).png
│   ├── XGBoost - ROC Curve.png
│   ├── XGBoost_best_model.pkl
│   ├── 모델학습결과.csv    
│   ├── data_preprocessing.csv  
│   ├── label_encoder.pkl
│   └── 보고서.pdf 
│   
└── README.md/                        
```
           
---

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

---

# 3. 🧼 데이터 전처리 과정 
## 3-1. 비정상 데이터 변환

- "Length of Stay" → 문자열 제거 후 정수형 변환
- NaN이 많은 열 (10,000건 이상 결측치) 제거
- 분석에 의미 없는 식별자 (의사 면허 번호, 우편번호 등) 제거 

## 3-2. 이상치 제거 & 희귀 범주 통합

- 수치형 컬럼: 정규분포 Z-score 기반 이상치 제거
- 범주형 컬럼: 빈도 낮은 범주(0.1% 미만)는 "Others" 통합

<br/><br/>

---

# 4. 🤖 모델링 및 학습 전략
## 4-1. 전처리 
<br/>
| 처리 항목     | 사용 기법                                            |
| --------- | ------------------------------------------------ |
| 범주형 인코딩   | Tree 기반 → OrdinalEncoder / Logistic Rregression 기반 → OneHotEncoder |
| 수치형 결측 대체 | KNN Imputer                                      |
| 정규화       | StandardScaler                                   |
| 클래스 불균형   | SMOTE (소수 클래스인 이탈 환자 과소표집 없이 오버샘플링)                      |
<br/>
<br/>
## 4-2. 모델 비교
<br/>
### 📊 모델별 성능 비교 결과

| Model             | Accuracy | Precision | Recall | F1 Score |
|------------------|----------|-----------|--------|----------|
| KNNNeighbors      | 0.9494   | 0.1681    | 0.3126 | 0.2186   |
| Decision Tree     | 0.9662   | 0.3003    | 0.3722 | 0.3324   |
| Random Forest     | 0.9841   | 0.9883    | 0.3010 | 0.4614   |
| Gradient Boosting | 0.9841   | 0.9911    | 0.2983 | 0.4586   |
| XGBoost           | 0.9831   | 0.8295    | 0.3206 | **0.4624** |


>> F1 Score 기준으로 가장 균형 잡힌 성능을 보인 **XGBoost**가 실용성과 성능 모두에서 우수하다고 인식.
 >> > 그 중 F1 Score 기준, XGBoost 모델이 가장 우수하여, XGBoost 모델을 선정. 

<br/><br/>

---

# 5. 선택한 모델 XGBoost 성능 높이기

* 이탈 환자 (LAMA) 여부를 예측하기 위해 여러 분류 모델(Logistic Regression, Random Forest, XGBoost 등)을 학습했다.
* 특히 이탈 환자는 전체 환자의 2.3% 내외로 극심한 클래스 불균형 문제를 가지고 있어, 일반적인 모델 학습 방법만으로는 좋은 성능을 얻기 어려운 상황이였다. 


> 3가지 전략을 통해 F1 Score를 극대화하는 방향으로 성능을 향상시키는 실험을 수행함.

<br/><br/>


## 5-1. Train Set, Test Set의 크기 확장

* 기본적으로 SMOTE는 Train set에만 적용되기 때문에, Test Set의 이탈 환자 수(y=1)가 너무 적어 모델 평가에 신뢰도가 떨어지는 문제가 있었다. 

> 기존 size를 1:1 유지하면서 키워가며 최적의 size 찾음\
> 사이즈 train = test = 0.02로 시작\
> 최대 30%까지 점진적으로 확장하면서 적절한 분할 비율을 탐색했고\
> Train : Test = 7:3 구성으로 안정적인 평가가 가능해졌다.
<br/>


## 5-2. 하이퍼파라미터 튜닝 (XGBoost 기준)

### 📊 Test Set Evaluation (Best XGBoost)

| Class | Precision | Recall | F1 Score | Support |
|-------|-----------|--------|----------|---------|
| 0 (정상 퇴원) | 0.98 | 1.00 | 0.99 | 48,510 |
| 1 (이탈 퇴원) | 0.89 | 0.32 | 0.47 | 1,123 |

| 지표 종류       | Precision | Recall | F1 Score |
|----------------|-----------|--------|----------|
| Accuracy        | -         | -      | **0.98** |
| Macro Average   | 0.94      | 0.66   | 0.73     |
| Weighted Average| 0.98      | 0.98   | 0.98     |

> ✅ **F1 Score (Test 기준): 0.4739**


<br/>
<br/>


## 5-3. Threshold 조정 

기본 threshold(0.5)에서의 F1 Score를 개선하기 위해 다양한 threshold를 실험. 

### 주요 실험 결과: 

| Threshold | Precision | Recall | F1 Score |
|-----------|-----------|--------|----------|
| 0.10      | 0.3005    | 0.6161 | 0.4040   |
| 0.15      | 0.4196    | 0.5336 | 0.4698   |
| 0.20      | 0.5387    | 0.4707 | 0.5024   |
| 0.25      | 0.6518    | 0.4257 | 0.5150   |
| 0.30      | 0.7565    | 0.3908 | **0.5153** ← 최적 F1
| 0.35      | 0.8343    | 0.3665 | 0.5093   |
| 0.40      | 0.8945    | 0.3488 | 0.5019   |
| 0.45      | 0.9372    | 0.3354 | 0.4940   |
| 0.50      | 0.9656    | 0.3263 | 0.4878   |


✅ **최적 Threshold = 0.30**,  
→ **F1 Score가 0.5153**으로 **기본보다 향상**됨
<br/>
<br/>

### F1 score 향상 전 XGBoost 성능 
| Model             | Accuracy | Precision | Recall | F1 Score |
|------------------|----------|-----------|--------|----------|
| XGBoost           | 0.9831   | 0.8295    | 0.3206 | **0.4624** |


### F1 score 향상 후 XGBoost 성능 
| Model             | Accuracy | Precision | Recall | F1 Score |
|------------------|----------|-----------|--------|----------|
| XGBoost           | 0.9831   | 0.7565    | 0.3908 | **0.5153** |


###  Test Set 확장, 파라미터 튜닝, Threshold 조정을 통해 F1 Score를 약 0.4624 → 0.5153 로 향상시킬 수 있었습니다.

>> 이는 실제 의료 시스템에서 실제 이탈 환자를 더 정확히 탐지하는 데에 기여할 수 있을 것이다.



💡 주요 인사이트

- LAMA는 **전체의 약 2%**밖에 되지 않아 클래스 불균형 문제를 반드시 고려해야 함

- Threshold 조정은 모델의 실용성과 경고 민감도 사이 균형을 맞추는 핵심 전략

- 병원이 이 모델을 실제로 활용할 경우, "최대한 많은 이탈 환자를 사전 탐지"하는 것이 중요
- → Precision이나 Recall은 활용에 따라 중요도가 달라지기 때문에, 우리팀은 F1 Score 기준으로 모델을 선정하였다. 


<br/>
<br/>

### 5-4. 튜닝 과정과 결과 시각화 

* 과정 
<img src="readme_img/XGB boost - F1 score vs Sample size.png>

* 결과 
<img src="readme_img/XGBoost-Threshold vs F1:precision:recall.png>

<img src="readme_img/XGB boost- Confusion Matrix.png>

<img src="readme_img/XGBoost - Precision Recall Curve .png>

<img src="readme_img/XGB boost- ROC Curve.png>




## 6. Feature Importances 파악하기 


<img src="readme_img/top 10 Featuree Importances(XGBoost).png>

### 환자 이탈 중요 요인
- 성별
- 나이
- 인종
- 외과적 수술 설명 후
- 병명 코드 (정신 질환, 약물/알코올 중독, HIV)
- 입원 기간

<br/><br/>

## 7.  주요 변수 영향도 및 이탈률 분석
Feature Importance 분석을 통해 이탈률에 가장 큰 영향을 주는 요인을 선별하고, 각 변수별 이탈 패턴을 구체적으로 해석

# 리드미 캡쳐가 들어야할거같아ㅏㅏㅏ 

<br/><br/>

## 8. 모델 활용성, 


## 9. 참고문헌 




---
# 💭 한 줄 회고

| 이름 | 한 줄 회고 |
|:------:|-------------------|
| 건우 | |
| 진슬 | 도망가지마.... 병원에 있어... |
| 진혁 | |
| 채우 | |




<br/><br/>
<br/><br/>
<br/><br/>


끝.










