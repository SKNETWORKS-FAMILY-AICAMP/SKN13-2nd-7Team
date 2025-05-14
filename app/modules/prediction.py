import streamlit as st
import numpy as np
import pickle

# 모델 불러오기 함수
@st.cache_resource
def load_model():
    # with open("model.pkl", "rb") as f:
    #     model = pickle.load(f)
    # return model
    return "model"


# 페이지 함수 - app.py에서 페이지 라우팅 할 때 사용
def show():
    #모델 로드
    model = load_model()

    # 변수 선언
    pred_result = 13.6

    # 페이지 title
    st.title("환자 이탈 가능성 예측")

    # 성별 선택 selectbox
    gender = st.selectbox(
        "성별",
        ("여성", "남성", "선택 안함"),
    )
    st.write(f"선택된 성별:{gender}")

    # 인종 선택 selectbox
    race = st.selectbox(
        "인종",
        ("백인", "흑인", "그 외 인종"),
    )
    st.write(f"선택된 성별:{race}")

    # 나이 선택 slider
    age = st.slider('나이', 1, 120, 30, 1)
    st.text(f'선택된 나이:{age}')

    # 출생 체중 선택 slider
    birth_weight = st.slider('출생 체중', 0.0, 4.5, 3.0, 0.1)
    st.text(f'선택된 출생 체중:{birth_weight}')

    st.button("예측하기")
    st.markdown(f"### 이탈 가능성 예측 결과:{pred_result}%")
