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

    return rare_summary

# 이상치 대체 함수
def replace_and_report_rare_categories(df, columns=None, threshold=0.001):
    """
    희귀 범주(비율 < threshold)를 'Others'로 대체하고, 컬럼별 'Others' 비율(%)을 반환합니다.

    Parameters:
    - df (pd.DataFrame): 원본 데이터프레임 (in-place로 수정됨)
    - columns (list): 대상 범주형 컬럼 리스트
    - threshold (float): 희귀 범주 판단 기준 비율 (기본: 0.001 = 0.1%)

    Returns:
    - pd.Series: 컬럼별 'Others' 비율 (%) 시리즈
    """
    if columns is None:
        raise ValueError("'columns' 인자에 범주형 컬럼 리스트를 지정해야 합니다.")
    
    ratio_dict = {}

    for col in columns:
        # 1. 희귀 범주 식별
        value_ratios = df[col].value_counts(normalize=True)
        rare_cats = value_ratios[value_ratios < threshold].index

        # 2. 희귀 범주 → Others로 대체 (in-place)
        df[col] = df[col].apply(lambda x: 'Others' if pd.notnull(x) and x in rare_cats else x)

        # 3. Others 비율 계산
        count_others = (df[col] == 'Others').sum()
        ratio = count_others / len(df)
        ratio_dict[col] = round(ratio * 100, 6)  # 소수점 6자리까지

    return pd.Series(ratio_dict, name='Others Ratio (%)')
