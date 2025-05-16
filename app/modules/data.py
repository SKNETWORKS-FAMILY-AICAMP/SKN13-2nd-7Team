import streamlit as st
import pandas as pd

def show():
    # 페이지 제목
    st.title("환자 데이터 통계")

    # 데이터 설명
    st.markdown("#### 데이터 정보")
    st.markdown("""
    2010년 뉴욕주의 병원 퇴원 환자 약 240만 건의 기록을 담고 있는 데이터셋입니다.  
    입원 사유, 진료비, 보험 유형, 퇴원 방식 등 다양한 의료 이용 정보를 포함하고 있으며,  
    환자의 특성과 병원 이용 패턴을 기반으로 이탈 가능성을 예측하는 데 활용됩니다.
    """)

    # 통계 분석 제목
    st.markdown("#### 환자 데이터 통계")

    # ============================
    # 기본 설정
    # ============================
    FEATHER_FILE_PATH = "data/hospital_discharges.feather"
    TABLE_HEIGHT = 600
    TABLE_WIDTH = 1200
    NUMERIC_COLUMNS = ['Total Charges', 'Total Costs', 'Length of Stay']

    # ============================
    # Feather 데이터 로드
    # ============================
    @st.cache_resource
    def load_data():
        return pd.read_feather(FEATHER_FILE_PATH)

    df = load_data()

    # ============================
    # 필터할 컬럼 선택
    # ============================
    selected_col = st.selectbox("🔍 필터할 컬럼 선택", df.columns)

    # ============================
    # 수치형 / 범주형 필터
    # ============================
    filtered_df = pd.DataFrame()

    if selected_col in NUMERIC_COLUMNS:
        min_val = float(df[selected_col].min())
        max_val = float(df[selected_col].max())
        selected_range = st.slider(
            f"'{selected_col}' 값 범위 선택",
            min_value=min_val,
            max_value=max_val,
            value=(min_val, max_val),
            step=(max_val - min_val) / 100
        )
        filtered_df = df[df[selected_col].between(*selected_range)]

    else:
        unique_vals = sorted(df[selected_col].dropna().unique())
        selected_val = st.selectbox(f"'{selected_col}'의 고유값 선택", unique_vals)
        filtered_df = df[df[selected_col] == selected_val]

    # ============================
    # 결과 출력
    # ============================
    st.markdown(f"**총 {len(filtered_df):,}개** 행이 필터링되었습니다.")
    st.dataframe(filtered_df, height=TABLE_HEIGHT, width=TABLE_WIDTH)
