# Titanic_Feature_Engineering

## 1. 프로젝트 개요

본 프로젝트는 Titanic 데이터셋을 활용하여 특성 공학(Feature Engineering)과 전처리 전략에 따른 머신러닝 모델의 성능 차이를 비교하고 분석하는 것을 목적으로 한다.

데이터 탐색부터 결측치 처리, 범주형 인코딩, 스케일링, 파생 변수 생성, 변수 선택, 모델 학습 및 평가까지 전체 머신러닝 파이프라인을 구현하였다.

## 2. 데이터셋

- 데이터셋: Titanic - Machine Learning from Disaster
- 전체 데이터 수: 891개
- 전체 컬럼 수: 12개
- 타겟 변수: Survived
- 학습 데이터: 712개
- 테스트 데이터: 179개
- 문제 유형: 이진 분류(Binary Classification)
- 데이터 출처: https://www.kaggle.com/competitions/titanic/data

## 3. 사용 기술 및 라이브러리

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- SHAP
- Optuna
- Jupyter Notebook

## 4. 주요 구현 내용

### EDA

- 결측치 및 결측치 비율 분석
- 이상치 탐색
- 변수별 분포 분석
- 상관관계 분석
- 타겟 변수 분포 분석
- Histogram, Boxplot, Heatmap, Countplot 시각화

### Feature Engineering

- 결측치 처리 방법 비교
- One-Hot Encoding 및 Label 방식 인코딩 비교
- StandardScaler, MinMaxScaler, RobustScaler 비교
- 파생 변수 6개 생성
- 중요도 기반 상위 10개 특성 선택
- 변수 선택 전후 성능 비교

### 모델 학습 및 평가

- Logistic Regression
- Random Forest
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

## 5. 실험 구성

| 실험 | 결측치 처리 | 인코딩 | 스케일링 | 변수 선택 |
|---|---|---|---|---|
| Base | 없음 | 없음 | 없음 | X |
| Exp-1 | Mean | One-Hot | Standard | X |
| Exp-2 | Median | Label | MinMax | O |
| Exp-3 | Most Frequent | One-Hot | Robust | O |

## 6. 추가 실험

- Scikit-learn Pipeline을 활용한 전처리 및 모델 학습
- GridSearchCV 기반 하이퍼파라미터 최적화
- SHAP 기반 모델 설명 가능성 분석
- Optuna 기반 자동 모델 탐색
- Feature Importance 시각화

## 7. 실험 결과 및 결론

특성 공학을 적용한 모델이 기본 모델보다 전반적으로 높은 성능을 보였다.

교차검증 결과를 기준으로 Exp-3의 Random Forest 모델을 선택하였으며, 테스트 데이터에서는 Exp-1의 Logistic Regression 모델이 높은 F1-score를 기록하였다.

이를 통해 전처리 및 변수 선택 전략의 효과는 모델의 특성에 따라 달라질 수 있으며, 적절한 특성 공학 과정이 모델의 예측 성능 향상에 기여한다는 것을 확인하였다.

## 8. 실행 방법

1. 저장소를 다운로드하거나 Git Clone을 수행한다.
2. requirements.txt에 명시된 라이브러리를 설치한다.
3. Jupyter Notebook을 실행한다.
4. notebooks 폴더의 Titanic_Feature_Engineering.ipynb 파일을 열고 전체 셀을 실행한다.

## 9. 참고 자료

- Kaggle Titanic Dataset
- Scikit-learn 공식 문서
- SHAP 공식 문서
- Optuna 공식 문서
