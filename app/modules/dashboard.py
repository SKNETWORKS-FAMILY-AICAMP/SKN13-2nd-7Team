import streamlit as st

def show():

    # 페이지 title
    st.title("데이터 시각화")

    # 탭 구성
    tab1, tab2, tab3, tab4 = st.tabs(["그래프 종류1", "그래프 종류2", "그래프 종류3", "그래프 종류4"])

    # 그래프 1
    with tab1:
        st.markdown("### 그래프 1번입니당")

    # 그래프 2
    with tab2:
        st.markdown("### 그래프 2번입니당")

    # 그래프 3
    with tab3:
        st.markdown("### 그래프 3번입니당")

    # 그래프 4
    with tab4:
        st.markdown("### 그래프 4번입니당")
