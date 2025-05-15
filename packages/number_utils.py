import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore

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
        plt.legend()

    plt.tight_layout()
    plt.show()


# 이상치 제거 함수
def remove_zscore_outliers_all(df, columns, z_thresh=2):
    """
    여러 컬럼에 대해 Z-score 기반 이상치를 포함한 행을 제거한 DataFrame을 반환합니다.

    Parameters:
    - df: pandas DataFrame
    - columns: 수치형 컬럼 리스트
    - z_thresh: 이상치 판단 기준 Z-score 값 (기본: 2)

    Returns:
    - pd.DataFrame: 이상치 제거 후 DataFrame
    """
    df_clean = df.copy()
    mask = pd.Series(False, index=df.index)

    for col in columns:
        data = df[col].dropna()
        zscores = zscore(data)
        outliers = (zscores > z_thresh) | (zscores < -z_thresh)
        outlier_idx = data[outliers].index
        mask[outlier_idx] = True

    removed_count = mask.sum()
    print(f"\n총 제거된 이상치 행 수: {removed_count}개")

    return df_clean[~mask]

