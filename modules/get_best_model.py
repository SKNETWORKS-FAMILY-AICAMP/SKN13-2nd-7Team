from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import Pipeline
import pickle
import os

def get_best_model(name, model, params, preprocessor, X_train, y_train):
    '''
    모델과 파라미터, 전처리기를 입력받아 최적의 모델을 찾는 함수
    최적 모델은 {model}_model.pkl 파일로 저장
    best parameter와 best score를 출력

    Args:
        model: 사용할 모델
        params: 모델의 파라미터
        preprocessor: 전처리기
        X_train: 학습 데이터
        y_train: 학습 레이블

    Returns:
        best_model: 최적 모델
        best_params(dict): 최적 파라미터
        best_score(dict): 최적 모델의 평가지표
    '''
    # 원-핫 인코딩을 사용하는 파이프라인 객체 생성 
    pipeline = Pipeline([
        ('preprocessor', preprocessor), 
        ('model', model)
    ])

    # RandomizedSearchCV 객체 생성
    rs = RandomizedSearchCV(
        pipeline,
        param_distributions=params,
        n_iter=10,
        scoring={'accuracy' : 'accuracy', 
                'precision' : 'precision', 
                'recall' : 'recall', 
                'f1' : 'f1'},
        refit='f1',
        cv=3,
        verbose=1,
        n_jobs=-1
    )

    # 모델 학습
    rs.fit(X_train, y_train)

    print("best parameter:", rs.best_params_)
    print("best score:", rs.best_score_) 

    # 최적 모델 저장
    best_model = rs.best_estimator_
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)
    with open(os.path.join(models_dir, f'{name}_model.pkl'), 'wb') as f:
        pickle.dump(best_model, f)

    # 최적 모델, 최적 파라미터, 평가지표 반환
    return best_model, rs.best_params_, rs.scoring