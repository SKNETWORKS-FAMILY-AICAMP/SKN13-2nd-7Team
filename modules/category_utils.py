import pandas as pd

def replace_rare_categories(df, cate_columns, threshold=0.001):
    """
    각 범주형 컬럼에서 등장 비율이 threshold 미만인 희귀 범주들을 'Others'로 대체합니다.

    Parameters:
    - df (pd.DataFrame): 처리할 원본 데이터프레임
    - cate_columns (list): 범주형 컬럼명을 담은 리스트
    - threshold (float): 희귀 범주의 기준이 되는 최소 등장 비율 (default 0.001 = 0.1%)

    Returns:
    - pd.DataFrame: 희귀 범주가 'Others'로 대체된 새로운 데이터프레임
    """
    df_copy = df.copy()
    
    for col in cate_columns:
        value_ratios = df_copy[col].value_counts(normalize=True)  # 비율 기준
        rare_cats = value_ratios[value_ratios < threshold].index  # 희귀 범주 추출
        df_copy[col] = df_copy[col].apply(lambda x: 'Others' if x in rare_cats else x)
    
    return df_copy


def calc_others_ratio(df, columns=None):
    """
    각 컬럼에서 'Others'라는 값이 전체 데이터에서 차지하는 비율(%)을 계산합니다.

    Parameters:
    - df (pd.DataFrame): 대상 데이터프레임
    - columns (list or None): 비율을 계산할 컬럼 리스트 (None이면 전체 컬럼 대상)

    Returns:
    - pd.Series: 컬럼별 'Others' 비율 (%)을 담은 시리즈
    """
    if columns is None:
        columns = df.columns  # 전체 컬럼 대상으로

    ratio_dict = {}
    for col in columns:
        if 'Others' in df[col].values:
            count_others = (df[col] == 'Others').sum()
            ratio = count_others / len(df)  # 전체 행 대비 Others 비율
            ratio_dict[col] = round(ratio * 100, 4)  # %로 변환
        else:
            ratio_dict[col] = 0.0  # Others 없음
    
    return pd.Series(ratio_dict, name='Others Ratio (%)')