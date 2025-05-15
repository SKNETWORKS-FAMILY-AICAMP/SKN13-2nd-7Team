import numpy as np
from imblearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -------------------------
# 모델 학습 클래스 정의
# -------------------------
class ModelTrainer:
    """
    다양한 분류 모델 학습 및 평가 클래스
    """

    def __init__(self, model, preprocessor=None):
        """
        Parameters:
        - model: 사용할 ML 모델 객체 (ex. RandomForestClassifier)
        - preprocessor: ColumnTransformer 전처리 파이프라인. preprocessor가 None이면 생략
        """
        steps = []
        if preprocessor:
            steps.append(('preprocessor', preprocessor))
        steps.append(('model', model))
        self.pipeline = Pipeline(steps)

    def fit(self, X_train, y_train):
        """모델 학습"""
        self.pipeline.fit(X_train, y_train)

    def evaluate(self, X_test, y_test, threshold=0.5):
        """
        테스트 데이터에 대해 예측 및 성능 평가
        
        Parameters:
        - X_test: 테스트 데이터
        - y_test: 정답값
        - threshold: 이진 분류 기준 임계값

        Returns:
        - dict: Accuracy, Precision, Recall, F1 Score
        """
        y_proba = self.pipeline.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)

        return {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1 Score': f1_score(y_test, y_pred, zero_division=0)
        }