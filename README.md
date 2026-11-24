# Titanic 생존 예측: Feature Engineering

## 1. 프로젝트 개요

Titanic 승객 데이터를 활용해 특성 공학(Feature Engineering)과 전처리 전략에 따른 머신러닝 분류 모델의 성능 차이를 비교·분석한 프로젝트입니다.

이 저장소는 **하나의 Jupyter Notebook으로 전체 과정을 실행할 수 있도록 구성**했습니다. 원본 학습 데이터(CSV)는 Notebook 내부에 압축된 형태로 포함되어 있으므로, 별도의 데이터 파일이나 Python 스크립트를 다운로드할 필요가 없습니다.

## 2. 저장소 구성

```text
Titanic-Feature-Engineering/
├── README.md
└── Titanic_Feature_Engineering.ipynb
```

- `Titanic_Feature_Engineering.ipynb`: 데이터 로드, EDA, 전처리, 파생 변수 생성, 모델 학습 및 평가, 추가 실험까지 포함한 전체 코드와 실행 결과

## 3. 데이터셋

- 데이터셋: Titanic - Machine Learning from Disaster
- 데이터 출처: [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic/data)
- 공개 CSV 참고 링크: [Titanic train.csv](https://github.com/agconti/kaggle-titanic/blob/master/data/train.csv)
- 데이터 크기: 891개 샘플, 12개 컬럼
- 예측 대상(Target): `Survived`
- 문제 유형: 이진 분류(Classification)

원본 `train.csv` 데이터는 Notebook 내부에 포함되어 있어 실행 시 별도 다운로드가 필요하지 않습니다.

## 4. 사용 기술

- Python
- Jupyter Notebook
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- XGBoost
- SHAP
- Optuna

## 5. 주요 구현 내용

### 5.1 데이터 탐색(EDA)

- 데이터 구조 및 크기 확인
- 결측치 개수와 비율 분석
- 이상치 탐색
- 변수별 분포 시각화
- 상관관계 분석
- 생존 여부(Target) 분포 확인
- Histogram, Boxplot, Heatmap, Countplot 시각화

### 5.2 특성 공학(Feature Engineering)

- 결측치 처리 방법 비교
- 범주형 변수 인코딩 방법 비교
- StandardScaler, MinMaxScaler, RobustScaler 비교
- 파생 변수 6개 생성
  - `FamilySize`
  - `IsAlone`
  - `FarePerPerson`
  - `HasCabin`
  - `Title`
  - `AgeGroup`

### 5.3 변수 선택(Feature Selection)

- Random Forest 기반 특성 중요도 분석
- 중요도 상위 변수 선택
- 변수 선택 전후 성능 비교

### 5.4 모델 학습 및 평가

- Logistic Regression
- Random Forest
- XGBoost
- Accuracy, Precision, Recall, F1-score, ROC-AUC 평가
- 교차검증 및 홀드아웃 테스트 데이터 평가

## 6. 비교 실험

과제에서 제시한 Base, Exp-1, Exp-2, Exp-3 조합을 구현했습니다.

| 실험 | 결측치 처리 | 인코딩 | 스케일링 | Feature Selection |
|---|---|---|---|---|
| Base | 없음 | 없음 | 없음 | 미적용 |
| Exp-1 | Mean | One-Hot | StandardScaler | 미적용 |
| Exp-2 | Median | Label 방식 | MinMaxScaler | 적용 |
| Exp-3 | Most Frequent | One-Hot | RobustScaler | 적용 |

또한 결측치 처리, 인코딩, 스케일링, 파생 변수, 변수 선택의 영향을 각각 살펴보는 독립 비교 실험을 포함했습니다.

## 7. 추가 실험

- Scikit-learn `Pipeline`을 이용한 전처리 및 모델 학습
- `GridSearchCV`를 이용한 하이퍼파라미터 탐색
- SHAP 기반 모델 예측 설명
- Optuna를 이용한 제한적 자동 모델·하이퍼파라미터 탐색(AutoML-lite)
- Feature Importance 시각화

## 8. 실행 방법

1. GitHub 저장소에서 `Titanic_Feature_Engineering.ipynb` 파일을 엽니다.
2. Jupyter Notebook 또는 Google Colab 환경에서 실행합니다.
3. 필요한 라이브러리가 설치되어 있지 않으면 Notebook의 **실행 환경** 셀에 있는 설치 명령을 실행합니다.
4. 위에서부터 순서대로 모든 셀을 실행하거나 `Run All`을 선택합니다.

실험이 실행되면 현재 작업 디렉터리에 `titanic_outputs/` 폴더가 생성되며, 데이터 사본과 EDA 그래프, 실험 결과 파일 등이 저장됩니다. 이 폴더는 실행 과정에서 자동 생성되므로 저장소에 별도로 업로드할 필요가 없습니다.

## 9. 재현성 및 유의사항

- 난수 시드는 `42`로 고정했습니다.
- 전처리와 변수 선택은 학습 데이터 및 교차검증 파이프라인 내부에서 수행하여 데이터 누수 가능성을 줄였습니다.
- 패키지 버전과 실행 환경에 따라 성능 지표에 소수점 수준의 차이가 발생할 수 있습니다.
- Base 실험은 결측치와 범주형 변수 전처리를 적용하지 않는 기준선으로, 원시 수치형 변수만 사용합니다.
- SHAP 결과는 모델의 예측 기여도를 설명하기 위한 것이며 변수와 생존 사이의 인과관계를 증명하지 않습니다.
- Optuna 실험은 본 프로젝트에서 설정한 탐색 범위 내의 제한적 AutoML-lite 실험입니다.

## 10. 참고 자료

- [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic/data)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [Optuna Documentation](https://optuna.org/)
