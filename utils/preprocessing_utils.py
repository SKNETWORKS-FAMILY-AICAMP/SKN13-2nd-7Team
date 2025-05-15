import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# -------------------------
# (1-1) 범주형 전처리 파이프라인 생성
# -------------------------
def create_categorical_pipeline(use_onehot=False):
    """
    범주형 인코딩 파이프라인 생성 함수

    Parameters:
    - use_onehot: True이면 OneHotEncoder, False이면 OrdinalEncoder 사용

    Returns:
    - 파이프라인 객체
    """
    if use_onehot:
        return make_pipeline(
            SimpleImputer(strategy='most_frequent'),
            OneHotEncoder(handle_unknown='ignore')
        )
    else:
        return make_pipeline(
            SimpleImputer(strategy='most_frequent'),
            OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        )

# -------------------------
# (1-2) 수치형 전처리 파이프라인 생성
# -------------------------
def create_numeric_pipeline():
    """
    수치형 컬럼에 대한 결측치 처리 + 표준화 파이프라인 생성

    Returns:
    - 파이프라인 객체
    """
    return make_pipeline(
        KNNImputer(n_neighbors=5),
        StandardScaler()
    )

# -------------------------
# (1-3) 전체 ColumnTransformer 구성
# -------------------------
def create_preprocessor(cate_columns, num_columns, use_onehot=False):
    """
    범주형 + 수치형 전처리 조합을 구성한 ColumnTransformer 반환

    Parameters:
    - cate_columns: 범주형 컬럼 리스트
    - num_columns: 수치형 컬럼 리스트
    - use_onehot: 범주형 인코딩 방식 선택

    Returns:
    - ColumnTransformer 객체
    """
    cat_pipe = create_categorical_pipeline(use_onehot=use_onehot)
    num_pipe = create_numeric_pipeline()

    return ColumnTransformer([
        ('cat', cat_pipe, cate_columns),
        ('num', num_pipe, num_columns)
    ])


# -------------------------
# (2) Train/Test 분할 함수
# -------------------------
def split_data(df, target_col='Patient Disposition', positive_class='Left Against Medical Advice',
               test_size=0.2, random_state=42):
    """
    이진 분류 문제를 위한 Train/Test 데이터 분할
    """
    X = df.drop(columns=[target_col])
    y = np.where(df[target_col].values == positive_class, 1, 0)
    return train_test_split(X, y, train_size=0.3, test_size=0.3, random_state=42, stratify=y)

# -------------------------
# (3) SMOTE 적용 함수 (train set만 적용)
# -------------------------
def apply_smote(X_train, y_train, sampling_strategy=0.1, k_neighbors=4, random_state=42):
    """
    훈련 데이터에만 SMOTE 오버샘플링을 적용
    """
    smote = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=k_neighbors, random_state=random_state)
    return smote.fit_resample(X_train, y_train)
