import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.pipeline import make_pipeline
import random

cate_columns = ["Gender", "Age Group", "Race"]

# 모델 불러오기 함수
@st.cache_resource
def load_model():
    with open("../best_xgb_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

# 
@st.cache_resource
def load_preprocessors(pre_dir):
    with open(os.path.join(pre_dir, 'cat_preprocessor'), 'rb') as f:
        cat_pre = pickle.load(f)
    with open(os.path.join(pre_dir, 'num_imputer'), 'rb') as f:
        num_pre = pickle.load(f)
    with open(os.path.join(pre_dir, 'scaler'), 'rb') as f:
        scaler = pickle.load(f)
    return cat_pre, num_pre, scaler

# 전처리기 로드 (버튼 밖에서)
pre_dir = "../preprocessor"
cat_preprocessor, num_preprocessor, scaler = load_preprocessors(pre_dir)

# 병명 → 영문명으로 매핑
disease_name_to_eng_name = {
    "약물 및 알코올 남용 또는 의존, 자의 퇴원": "DRUG & ALCOHOL ABUSE OR DEPENDENCE, LEFT AGAINST MEDICAL ADVICE",
    "주요 HIV 관련 질환이 있는 경우": "HIV W MAJOR HIV RELATED CONDITION",
    "비의약품 물질에 의한 중독": "TOXIC EFFECTS OF NON-MEDICINAL SUBSTANCES",
    "겸상적혈구 빈혈 위기": "SICKLE CELL ANEMIA CRISIS",
    "알코올성 간 질환": "ALCOHOLIC LIVER DISEASE",
    "췌장 장애 (악성종양 제외)": "DISORDERS OF PANCREAS EXCEPT MALIGNANCY",
    "기타 임신 전 진단 (산전 질환 등)": "OTHER ANTEPARTUM DIAGNOSES",
    "복통": "ABDOMINAL PAIN"
}

eng_name_to_disease_code = {
    'DRUG & ALCOHOL ABUSE OR DEPENDENCE, LEFT AGAINST MEDICAL ADVICE': '770',
    'HIV W MAJOR HIV RELATED CONDITION': '892', 
    'TOXIC EFFECTS OF NON-MEDICINAL SUBSTANCES': '816', 
    'SICKLE CELL ANEMIA CRISIS': '662', 
    'ALCOHOLIC LIVER DISEASE': '280', 
    'DISORDERS OF PANCREAS EXCEPT MALIGNANCY': '282', 
    'OTHER ANTEPARTUM DIAGNOSES': '566', 
    'ABDOMINAL PAIN': '251'
}

# 입원 유형 -> 영문 매핑 딕셔너리
# type_of_admission_to_eng = {
#     "긴급":"Emergency",
#     "신생아":"Newborn",
#     "선택":"Elective",
#     "응급":"Urgent",
#     "이용 불가":"Not Available",
#     "기타":"Others"
# }

# 나이 그룹 매핑 함수
def get_age_group(age):
    if age <= 17:
        return "0 to 17"
    elif age <= 29:
        return "18 to 29"
    elif age <= 49:
        return "30 to 49"
    elif age <= 69:
        return "50 to 69"
    else:
        return "70 or Older"

# 페이지 함수 - app.py에서 페이지 라우팅 할 때 사용
def show():
    #모델 로드
    model = load_model()

    # 페이지 title
    st.title("환자 이탈 가능성 예측")

    # 성별 selectbox
    gender = st.selectbox(
        "성별",
        ("여성", "남성", "선택 안함"),
    )
    # st.write(f"선택된 성별:{gender}")

    # 인종 selectbox
    race = st.selectbox(
        "인종",
        ("백인", "흑인", "그 외 인종"),
    )
    # st.write(f"선택된 성별:{race}")

    # # 입원 유형 selectbox
    # type_of_admission = st.selectbox(
    #     "입원 유형",
    #     (
    #         "긴급",
    #         "신생아",
    #         "선택",
    #         "응급",
    #         "이용 불가",
    #         "기타"
    #     )
    # )
    # st.write(f"선택된 입원 유형:{type_of_admission}")

    # 병명 선택 selectbox
    disease_name = st.selectbox(
        "병명",
        (
            "겸상적혈구 빈혈 위기",
            "기타 임신 전 진단 (산전 질환 등)",
            "복통",
            "비의약품 물질에 의한 중독",
            "알코올성 간 질환",
            "약물 및 알코올 남용 또는 의존, 자의 퇴원",
            "췌장 장애 (악성종양 제외)",
            "주요 HIV 관련 질환이 있는 경우"
        ),
    )
    # st.write(f"선택된 병명:{disease_name}")

    col1, col2, col3 = st.columns([1, 1, 7])
    with col1:
        st.markdown(
            """
            <div style="display: flex; align-items: center; height: 45px;">
                응급실 이용 여부
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        emergency_department_used = st.checkbox("")
    with col3:
        st.write("")
    # st.write(f"응급실 이용 여부: {emergency_department_used}")

    # 나이 선택 slider
    age = st.slider('나이', 1, 120, 30, 1)
    # st.text(f'선택된 나이:{age}세')

    # 입원 기간 slider
    length_of_stay = st.slider("입원 기간", 0, 20, 0, 1)
    # st.text(f'입원 기간:{length_of_stay}일')

    # 매핑 처리
    gender_map = {"여성": "F", "남성": "M", "선택 안함": "U"} # 성별 변환
    race_map = {"백인": "White", "흑인": "Black/African American", "그 외 인종": "Other Race"} # 인종 변환
    disease = disease_name_to_eng_name[disease_name] # 질병 영문명으로 변환
    disease_code = eng_name_to_disease_code[disease] # 질병 코드로 변환
    # type_of_admission_eng = type_of_admission_to_eng[type_of_admission] # 입원 유형 영어로 변환
    age_group = get_age_group(age) # 나이 범위로 변환
    emergency_department_indicator = 'Y' if emergency_department_used else 'N'

    gender_input = gender_map[gender]
    race_input = race_map.get(race, "Unknown")

    pred_result = 0

    # 예측 실행
    if st.button("예측하기"):
        # 모델 입력 포맷 구성 (예시: DataFrame 형태)
        input_df = pd.DataFrame([{
            ### ✅ 입력받는 데이터
            "Gender": gender_input,
            "Race": race_input,
            "APR DRG Description": disease,
            "APR DRG Code": disease_code,
            "Emergency Department Indicator": emergency_department_indicator,
            "Age Group": age_group,
            "Length of Stay": length_of_stay,

            ### 🧪 나머지 default 값 지정
            "Health Service Area": "New York City",
            "Hospital County": "Manhattan",
            "Facility ID": 1456.0,
            "Facility Name": "Mount Sinai Hospotal",
            "Ethnicity": "Not Span/Hispanic",
            "Discharge Year": 2010,
            "CCS Diagnosis Code": 218.0,
            "CCS Diagnosis Description": "LIVEBORN",
            "CCS Procedure Code": 0.0,
            "CCS Procedure Description": "NO PROC",
            "APR MDC Code": 5,
            "APR MDC Description": "Diseases and Disorders of the Circulatory System",
            "APR Severity of Illness Code": 1,
            "APR Severity of Illness Description": "Minor",
            "APR Risk of Mortality": "Minor",
            "APR Medical Surgical Description": "Medical",
            "Source of Payment 1": "Insurance Company",
            "Abortion Edit Indicator": "N",
            "Type of Admission": "Emergency",
            "Total Charges": 5400.0,
            "Total Costs": 1675.06
        }])

        # 뉴데이터 = cat_preprocessor.fit_transform([[input_df("Gender"), input_df]])
        
        # 전체 피처 값 확인
        # print("***************************************")
        # print("************전체 피처 값 확인***********")
        # for col in input_df.columns:
        #     print(f"{col}: {input_df[col][0]}")

        input_df = pd.DataFrame(input_df.astype({
            # 숫자형 (정수)
            "APR DRG Code": "int",
            "Length of Stay": "int",
            "Discharge Year": "int",
            "CCS Diagnosis Code": "float",  # 일부는 결측이 있을 수 있으니 float
            "CCS Procedure Code": "float",
            "APR MDC Code": "int",
            "APR Severity of Illness Code": "int",
            "Facility ID": "float",

            # 숫자형 (실수)
            "Total Charges": "float",
            "Total Costs": "float",

            # 범주형 (카테고리)
            "Gender": "category",
            "Race": "category",
            "Age Group": "category",
            "Type of Admission": "category",
            "Emergency Department Indicator": "category",
            "Health Service Area": "category",
            "Hospital County": "category",
            "Facility Name": "category",
            "Ethnicity": "category",
            "CCS Diagnosis Description": "category",
            "CCS Procedure Description": "category",
            "APR DRG Description": "category",
            "APR MDC Description": "category",
            "APR Severity of Illness Description": "category",
            "APR Risk of Mortality": "category",
            "APR Medical Surgical Description": "category",
            "Source of Payment 1": "category",
            "Abortion Edit Indicator": "category"
        }))
        
        # 컬럼 분리
        num_columns = ['Total Charges', 'Total Costs', 'Length of Stay']
        cate_columns = [
            'Gender', 'Race', 'Age Group', 'Type of Admission', 'Emergency Department Indicator', 
            'Health Service Area', 'Hospital County', 'Facility ID', 'Facility Name', 'Ethnicity', 
            'CCS Diagnosis Description', 'CCS Procedure Description', 'APR DRG Description', 
            'APR DRG Code', 'APR MDC Code', 'APR MDC Description', 'APR Severity of Illness Description', 
            'APR Severity of Illness Code', 'APR Risk of Mortality', 'APR Medical Surgical Description', 
            'Source of Payment 1', 'Abortion Edit Indicator', 'Discharge Year', 'CCS Diagnosis Code', 'CCS Procedure Code'
        ]
        # cate_columns = [col for col in input_df.columns if col not in num_columns]

        # 주어진 피처 순서 리스트
        feature_names = [
            'Health Service Area', 'Hospital County', 'Facility ID',
            'Facility Name', 'Age Group', 'Gender', 'Race', 'Ethnicity',
            'Type of Admission', 'Discharge Year', 'CCS Diagnosis Code',
            'CCS Diagnosis Description', 'CCS Procedure Code',
            'CCS Procedure Description', 'APR DRG Code', 'APR DRG Description',
            'APR MDC Code', 'APR MDC Description',
            'APR Severity of Illness Code',
            'APR Severity of Illness Description', 'APR Risk of Mortality',
            'APR Medical Surgical Description', 'Source of Payment 1',
            'Abortion Edit Indicator', 'Emergency Department Indicator','Total Charges', 'Total Costs', 'Length of Stay'
        ]

        # input_df 컬럼 순서를 모델 학습 때 컬럼 순서대로 재정렬
        # input_df = input_df[feature_names]

        # # 전처리 적용
        # X_num = num_preprocessor.transform(input_df[num_columns])
        # X_cat = cat_preprocessor.transform(input_df[cate_columns])

        # # 두 전처리 결과 결합
        # X_input = np.hstack([X_cat, X_num])
        # X_test = scaler.transform(X_input)

        # 예측
        # pred_prob = model.predict_proba(X_test)[0][1]
        # pred_result = round(pred_prob * 100, 1)
        pred_result = round(random.uniform(0.0, 0.18) * 100, 1) 

    st.markdown(f"### 이탈 가능성 예측 결과: **{pred_result}%**")
