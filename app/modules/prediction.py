import streamlit as st
import numpy as np
import pandas as pd
import pickle


cate_columns = ["Gender", "Age Group", "Race"]

# 모델 불러오기 함수
@st.cache_resource
def load_model():
    with open("../best_xgb_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

# 병명 → 질병 코드 매핑 딕셔너리
disease_name_to_code = {
    "알코올 및 약물 남용 또는 의존": 770,
    "의학/정신과 영역 알코올·약물 남용/의존": 894,
    "기타 위장관 질환": 816,
    "정신병": 892,
    "신장 및 요로 감염": 662,
    "지적 장애": 890,
    "피부·피하조직·유방 손상": 280,
    "근골격계 및 결합조직 손상": 282,
    "기타 산전 진단": 566,
    "비세균성 위장염·구토·설사": 251,
}

# 입원 유형 -> 영문 매핑 딕셔너리
type_of_admission_to_eng = {
    "긴급":"Emergency",
    "신생아":"Newborn",
    "선택":"Elective",
    "응급":"Urgent",
    "이용 불가":"Not Available",
    "기타":"Others"
}

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
    st.write(f"선택된 성별:{gender}")

    # 인종 selectbox
    race = st.selectbox(
        "인종",
        ("백인", "흑인", "그 외 인종"),
    )
    st.write(f"선택된 성별:{race}")

    # 입원 유형 selectbox
    type_of_admission = st.selectbox(
        "입원 유형",
        (
            "긴급",
            "신생아",
            "선택",
            "응급",
            "이용 불가",
            "기타"
        )
    )
    st.write(f"선택된 입원 유형:{type_of_admission}")

    # 병명 선택 selectbox
    disease_name = st.selectbox(
        "병명",
        (
            "근골격계 및 결합조직 손상",
            "비세균성 위장염·구토·설사",
            "신장 및 요로 감염",
            "알코올 및 약물 남용 또는 의존",
            "의학/정신과 영역 알코올·약물 남용/의존",
            "지적 장애",
            "정신병",
            "피부·피하조직·유방 손상",
            "기타 산전 진단",
            "기타 위장관 질환"
        ),
    )
    st.write(f"선택된 병명:{disease_name}")

    # 나이 선택 slider
    age = st.slider('나이', 1, 120, 30, 1)
    st.text(f'선택된 나이:{age}세')

    # 입원 기간 slider
    length_of_stay = st.slider("입원 기간", 0, 20, 0, 1)
    st.text(f'입원 기간:{length_of_stay}일')

    # # 총 비용 slider
    # total_costs = st.slider("총 비용($)", 0, 126559, 63000, 1)
    # st.text(f'총 비용:{total_costs}$')

    # # 총 청구 금액 slider
    # total_charges = st.slider("총 청구 금액($)", 0, 55346, 25000, 1)
    # st.text(f'총 청구 금액:{total_costs}$')


    # 매핑 처리
    gender_map = {"여성": "F", "남성": "M", "선택 안함": "U"} # 성별 변환
    race_map = {"백인": "White", "흑인": "Black/African American", "그 외 인종": "Other Race"} # 인종 변환
    disease_code = disease_name_to_code[disease_name] # 질병 코드로 변환
    type_of_admission_eng = type_of_admission_to_eng[type_of_admission] # 입원 유형 영어로 변환
    age_group = get_age_group(age) # 나이 범위로 변환

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
            "APR DRG Code": disease_code,
            "Type of Admission": type_of_admission_eng,
            "Age Group": age_group,
            "Length of Stay": length_of_stay,

            ### 🧪 나머지 default 값 지정 (수정 필요 !!!)
            "Total Costs": "total_costs",
            "Total Charges": "total_charges",
            "Health Service Area": "Unknown",
            "Hospital County": "Unknown",
            "Facility ID": "0000",
            "Facility Name": "Unknown Facility",
            "Ethnicity": "Unknown",  # 예: "Not Spanish/Hispanic"
            "Patient Disposition": "Other",
            "Discharge Year": 2020,
            "CCS Diagnosis Code": 0,
            "CCS Diagnosis Description": "Unknown",
            "CCS Procedure Code": 0,
            "CCS Procedure Description": "None",
            "APR DRG Description": "Unknown",
            "APR MDC Code": 0,
            "APR MDC Description": "Unknown",
            "APR Severity of Illness Code": 1,
            "APR Severity of Illness Description": "Minor",
            "APR Risk of Mortality": "Minor",
            "APR Medical Surgical Description": "Medical",
            "Source of Payment 1": "Self Pay",
            "Abortion Edit Indicator": "N",
            "Emergency Department Indicator": "N"
            }])
        
        # 전체 피처 값 확인
        print("***************************************")
        print("************전체 피처 값 확인***********")
        for col in input_df.columns:
            print(f"{col}: {input_df[col][0]}")

        # 여기가 핵심: 각 column의 타입을 명시적으로 지정
        input_df = input_df.astype({
            "Gender": "category",
            "Race": "category",
            "APR DRG Code": "int",
            "Type of Admission": "category",
            "Age Group": "category",
            "Length of Stay": "int"
        })

        pred_prob = model.predict_proba(input_df)[0][1]
        pred_result = round(pred_prob * 100, 1)
    
    # 임시 결과 - 수정 필요 !!!!
    # pred_result = 13.6
    # st.button("예측하기")
    st.markdown(f"### 이탈 가능성 예측 결과: **{pred_result}%**")
