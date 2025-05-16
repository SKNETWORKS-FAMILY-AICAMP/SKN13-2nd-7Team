import pandas as pd

def preprocess_and_drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    주어진 원본 DataFrame(df)에서 다음 전처리 기준에 따라 컬럼을 제거한 새로운 DataFrame을 반환합니다.

    전처리 기준:
    1. 'Length of Stay' 컬럼의 비정상값을 NaN으로 처리 후 NaN 행 제거 및 정수형으로 변환
    2. NaN 값이 10,000개를 초과하는 컬럼 제거
    3. 의미 없는 컬럼 제거: 우편번호, 인덱스, 주치의 면허번호, 운영 인증번호
    4. 몸무게가 0인 값이 대부분인 'Birth Weight' 컬럼 제거

    Parameters
    ----------
    df : pd.DataFrame
        전처리를 수행할 원본 데이터프레임

    Returns
    -------
    pd.DataFrame
        지정된 컬럼들이 제거된 전처리된 데이터프레임
    """

    # 1. 'Length of Stay' 컬럼 처리
    df['Length of Stay'] = pd.to_numeric(df['Length of Stay'], errors='coerce')
    df = df.dropna(subset=['Length of Stay'])
    df['Length of Stay'] = df['Length of Stay'].astype(int)

    # 2. NaN 개수가 10,000개 초과인 컬럼 제거
    nan_columns = df.columns[df.isna().sum() > 10_000]
    df = df.drop(columns=nan_columns)

    # 3. 의미 없는 컬럼 제거
    drop_columns = [
        'Zip Code - 3 digits',
        'index',
        'Attending Provider License Number',
        'Operating Certificate Number',
        'Birth Weight'
    ]
    df = df.drop(columns=[col for col in drop_columns if col in df.columns])

    return df
