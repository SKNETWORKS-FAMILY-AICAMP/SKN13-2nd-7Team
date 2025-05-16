import streamlit as st

# 반드시 첫 Streamlit 명령어로 호출
st.set_page_config(page_title="환자 이탈 예측 시스템", layout="wide")

# 이후 일반 모듈 import
import pandas as pd
import pickle
from streamlit_option_menu import option_menu

# 그 다음 내부 모듈 import
from modules import about, data, prediction, dashboard

with st.sidebar:
    selected = option_menu(
        menu_title="사이드바 이름수정",
        options= ['About', 'Data', 'Prediction', 'Dashboard'],
        # icons=[],
        default_index=0, # 선택 인덱스 디폴트값
    )

if selected == "About":
    about.show()
elif selected == "Data":
    data.show()
elif selected == "Prediction":
    prediction.show()
elif selected == "Dashboard":
    dashboard.show()
