import streamlit as st

def show():
    st.title("🏥 환자 이탈 예측 프로젝트")
    st.markdown("### Left Against Medical Advice (LAMA) Prediction")

    st.markdown("""
    병원에서는 환자의 퇴원 방식(Patient Disposition)에 따라 **수익성과 운영 효율성**에 큰 영향을 받습니다.  
    그 중에서도 **의사 권고 없이 자의로 퇴원하는 경우 (LAMA)** 는 병원에 다음과 같은 부정적인 영향을 줄 수 있습니다:
    
    - 병상 회전율 저하
    - 의료 품질 지표 악화
    - 병원 수익 손실


    본 프로젝트는 뉴욕주의 병원 퇴원 환자 데이터(약 240만 건)를 활용하여,  
    **이탈 환자(=자의 퇴원 환자)를 사전에 예측하는 머신러닝 분류 모델**을 구축하는 것을 목표로 합니다.
    """)

    st.markdown("## 🎯 프로젝트 목표")
    st.info("**이탈 환자 예측 모델 개발 (Binary Classification)**")

    st.markdown("""
    - 🔍 데이터셋: New York State Hospital Discharge Dataset (2010)
    - 🧪 이진 분류 목표: 자의 퇴원 (`class 1`) vs 정상 퇴원 (`class 0`)
    - 🧠 사용 모델: Logistic Regression, Random Forest, XGBoost 등
    """)

    st.markdown("---")

    st.markdown("## 📊 주요 인사이트")
    st.markdown("""
    - 이탈률이 높은 진단군은 다음과 같으며, 대부분 정신과/약물 관련 질환이 포함됩니다:

        | 순위 | 진단군 (한글)                           | 이탈률   |
        |------|--------------------------------------|--------|
        | 1    | 알코올 및 약물 남용 또는 의존              | 100%   |
        | 2    | 의학/정신과 영역 알코올·약물 남용/의존       | 13.9%  |
        | 3    | 기타 위장관 질환                        | 13.4%  |
        | 4    | 정신병                              | 11.8%  |
        | 5    | 신장 및 요로 감염                       | 10.7%  |

    - 이탈 환자는 **소수 클래스**이며, recall을 높이는 것이 실질적인 의료 운영 측면에서 중요합니다.
    - 최종 선정 모델: **XGBoost** → F1 Score 기준 가장 균형 잡힌 성능

    """)

    st.success("🔔 좌측 메뉴에서 데이터 통계, 모델 결과, 시각화, FAQ 등 다양한 내용을 확인해보세요!")

