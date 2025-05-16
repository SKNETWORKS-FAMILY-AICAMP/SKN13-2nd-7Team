import streamlit as st

def show():
    # 페이지 title
    st.title("데이터 시각화")

    # 탭 구성
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["성별", "연령대", "응급실 방문 여부", "인종", "입원일수", "APR 입원유형", "DRG 코드"])

     # 그래프 1
    with tab1:
        st.image("../images/성별에 따른 이탈률.png")

    # 그래프 2
    with tab2:
        st.image("../images/연령대별 환자 이탈률.png")

    # 그래프 3
    with tab3:
        st.image("../images/응급실 방문 여부에 따른 이탈률.png")

    # 그래프 4
    with tab4:
        st.image("../images/인종별 환자 이탈률.png")

    # 그래프 5
    with tab5:
        st.image("../images/입원일수별 환자 이탈률.png")

    # 그래프 6
    with tab6:
        st.image("../images/APR 입원유형별 환자 이탈률.png")

    # 그래프 7
    with tab7:
        st.image("../images/DRG 코드별 환자 이탈률.png")