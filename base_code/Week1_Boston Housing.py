# Boston Housing — Linear Regression & Coefficient Analysis

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.datasets import fetch_openml

# 1. 데이터 로딩 (OpenML에서 Boston 데이터셋 불러오기)
boston = fetch_openml(name="boston", version=1, as_frame=True)
df = boston.frame   # 13개 특성 + MEDV(목표값)

X = df.drop(columns=["MEDV"])
y = df["MEDV"]
feature_names = X.columns

print(X.shape, y.shape)
df.head()

# ===========================================================
# 아래부터는 TODO: 직접 작성할 부분
# ===========================================================

# 2. 간단 EDA
# - 결측치 확인하기 (X.isna().sum())
# - describe()로 기본 통계 확인하기
# - MEDV(target)의 분포를 히스토그램으로 시각화해보면 좋음

# 3. 선형회귀 모델 학습
# - StandardScaler와 LinearRegression을 Pipeline으로 묶어서 사용
#   예: make_pipeline(StandardScaler(), LinearRegression())
# - model.fit(X, y) 호출

# 4. 회귀계수 확인
# - model.named_steps["linearregression"].coef_ 로 계수 가져오기
# - pd.DataFrame으로 feature와 coef 묶어서 정리
# - 절댓값이 큰 순서대로 정렬해서 어떤 변수가 영향력이 큰지 보기

# 5. 긍정적/부정적 영향 확인
# - coef가 가장 큰 특성 → 가격 상승에 가장 긍정적 영향
# - coef가 가장 작은 특성 → 가격 하락에 가장 큰 부정적 영향
# - print()로 결과 출력

# 6. 시각화
# - matplotlib.pyplot.bar() 로 feature별 coef 시각화
# - plt.axhline(0) 그려서 양/음 구분하기
# - X축: feature 이름, Y축: coefficient

# 7. 추가 분석 (선택)
# - coef_df를 CSV로 저장하기
# - R², RMSE 등 성능 지표 계산해보기
# - Ridge/Lasso와 비교해보기
# 7. 추가 분석 과정을 안하고, 6번까지 하고 인증하셔도 괜찮습니다. 
