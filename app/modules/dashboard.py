import streamlit as st

def show():

    image_path1 = "../images/Num_Columns_Outliers.png"
    image_path2 = ""
    image_path3 = ""
    image_path4 = ""

    # 페이지 title
    st.title("데이터 시각화")

    # 탭 구성
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["그래프 종류1", "그래프 종류2", "그래프 종류3", "그래프 종류4", "그래프 종류5", "그래프 종류6", "그래프 종류7"])

    # 그래프 1
    with tab1:
        st.markdown("1번 그래프 설명")
        st.image("../images/Num_Columns_Outliers.png", caption="예시 이미지1")

    # 그래프 2
    with tab2:
        st.markdown("2번 그래프 설명")
        st.image("../images/PR_Curve.png", caption="예시 이미지2")

    # 그래프 3
    with tab3:
        st.markdown("3번 그래프 설명")
        st.image("../images/radar_chart.png", caption="예시 이미지3")

    # 그래프 4
    with tab4:
        st.markdown("4번 그래프 설명")
        st.image("../images/ROC_Curve.png", caption="예시 이미지4")

    # 그래프 5
    with tab4:
        st.markdown("5번 그래프 설명")
        st.image("../images/ROC_Curve.png", caption="예시 이미지5")

    # 그래프 6
    with tab4:
        st.markdown("6번 그래프 설명")
        st.image("../images/ROC_Curve.png", caption="예시 이미지6")

    # 그래프 7
    with tab4:
        st.markdown("7번 그래프 설명")
        st.image("../images/ROC_Curve.png", caption="예시 이미지7")
