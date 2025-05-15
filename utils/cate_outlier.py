import pandas as pd

# 이상치 확인 함수
def preview_rare_categories(df, columns, threshold=0.001):
    """
    각 범주형 컬럼에서 threshold 미만 비율로 등장하는 희귀 범주들을 출력합니다.

    Parameters:
    - df (pd.DataFrame): 대상 데이터프레임
    - columns (list): 확인할 컬럼 리스트
    - threshold (float): 희귀 범주 판단 기준 비율

    Returns:
    - dict: {컬럼명: 희귀값 리스트} 형태의 딕셔너리 (없으면 빈 리스트)
    """
    if columns is None:
        raise ValueError("'columns' 인자에 범주형 컬럼 리스트를 지정해야 합니다.")

    rare_summary = {}

    for col in columns:
        value_ratios = df[col].value_counts(normalize=True)
        rare_cats = value_ratios[value_ratios < threshold].index.tolist()
        rare_summary[col] = rare_cats
        
        if rare_cats:
            print(f"'{col}' 컬럼에서 희귀 범주 발견 ({len(rare_cats)}개): {rare_cats}")
        else:
            print(f"'{col}' 컬럼에는 희귀 범주 없음")


# 이상치 대체 함수
def replace_rare_categories(df, columns, threshold=0.001):
    """
    컬럼별 희귀 범주(비율 < threshold)를 'Others'로 대체한 DataFrame을 반환합니다.

    Parameters:
    - df (pd.DataFrame): 원본 데이터프레임
    - columns (list): 희귀 범주 처리 대상 범주형 컬럼 리스트
    - threshold (float): 희귀 범주 판단 기준 비율

    Returns:
    - pd.DataFrame: 희귀 범주가 'Others'로 대체된 새로운 DataFrame
    """
    if columns is None:
        raise ValueError("'columns' 인자에 범주형 컬럼 리스트를 지정해야 합니다.")

    df_new = df.copy()

    for col in columns:
        df_new[col] = df_new[col].astype(str)
        value_ratios = df_new[col].value_counts(normalize=True)
        rare_cats = value_ratios[value_ratios < threshold].index
        df_new[col] = df_new[col].apply(lambda x: 'Others' if x in rare_cats else x)

    return df_new
