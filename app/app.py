import streamlit as st
import pandas as pd
import pickle
from streamlit_option_menu import option_menu

from modules import about, data, prediction, dashboard

# 페이지 설정
st.set_page_config(page_title="환자 이탈 예측 시스템", layout="wide")



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
