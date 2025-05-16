import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore
import os
img_dir = "images"
os.makedirs(img_dir, exist_ok=True)

# 이상치 시각화 함수
def plot_zscore_outliers_all(df, columns, bins_list, z_thresh=2):
    """
    여러 컬럼에 대해 Z-score 기반 이상치를 시각화하고 이상치 수를 출력합니다.

    Parameters:
    - df: pandas DataFrame
    - columns: 수치형 컬럼 리스트
    - bins_list: 각 컬럼별 히스토그램 bin 수 리스트
    - z_thresh: 이상치 기준 Z-score 값 (기본: 2)
    """
    assert len(columns) == len(bins_list), "columns와 bins_list의 길이가 같아야 합니다."

    plt.figure(figsize=(12, 8))

    for idx, (col, bin_count) in enumerate(zip(columns, bins_list)):
        data = df[col].dropna()
        zscores = zscore(data)
        outliers = (zscores > z_thresh) | (zscores < -z_thresh)
        outlier_count = outliers.sum()

        print(f"'{col}' 컬럼 이상치 수 (|Z| > {z_thresh}): {outlier_count}개")

        plt.subplot(2, 2, idx + 1)
        plt.hist(zscores, bins=bin_count, color='skyblue', edgecolor='black')
        plt.axvline(z_thresh, color='red', linestyle='--', label=f'Z = +{z_thresh}')
        plt.axvline(-z_thresh, color='red', linestyle='--', label=f'Z = -{z_thresh}')
        plt.title(f"{col} (Outliers: {outlier_count})")
        plt.xlabel("Z-score")
        plt.ylabel("Count")
        plt.xlim(-5, 5)
        plt.grid(True)
        plt.savefig(os.path.join(img_dir, f"zscore_outliers_{col}.png"))
        plt.legend()

    plt.tight_layout()
    plt.show()


# 이상치 제거 함수
def remove_zscore_outliers_all(df, columns, z_thresh=2):
    """
    Z-score 기반 이상치를 포함한 행을 제거한 DataFrame을 반환합니다.

    Parameters:
    - df (pd.DataFrame): 원본 데이터프레임
    - columns (list): 수치형 컬럼 리스트
    - z_thresh (float): 이상치 판단 기준 Z-score 값 (기본값: 2)

    Returns:
    - pd.DataFrame: 이상치 제거 후의 새로운 DataFrame
    """
    if columns is None:
        raise ValueError("'columns' 인자에 수치형 컬럼 리스트를 지정해야 합니다.")
    
    df_new = df.copy()
    mask = pd.Series(False, index=df.index)

    for col in columns:
        zscores = zscore(df_new[col].dropna())
        outlier_idx = df_new[col].dropna().index[(zscores > z_thresh) | (zscores < -z_thresh)]
        mask[outlier_idx] = True

    return df_new[~mask]
