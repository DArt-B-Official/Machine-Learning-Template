# Boston Housing — Linear Regression & Coefficient Analysis

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.datasets import fetch_openml

# 1. 데이터 로딩
# OpenML에서 Boston 데이터 불러오기
boston = fetch_openml(name="boston", version=1, as_frame=True)
df = boston.frame  # 13개 특성 + MEDV(타깃)

X = df.drop(columns=["MEDV"])
y = df["MEDV"]
feature_names = X.columns

print(X.shape, y.shape)
df.head()

# 2. 간단 확인
print("결측치 개수:\n", X.isna().sum(), "\n")
print(df.describe().T)

# 3. 학습: 표준화 + 선형회귀
model = make_pipeline(
    StandardScaler(with_mean=True, with_std=True),
    LinearRegression()
)
model.fit(X, y)

lin = model.named_steps["linearregression"]
coef = lin.coef_
intercept = lin.intercept_

coef_df = pd.DataFrame({"feature": feature_names, "coef": coef})
coef_df["abs_coef"] = coef_df["coef"].abs()

# 계수 크기 확인
coef_sorted_pos = coef_df.sort_values("coef", ascending=False)
coef_sorted_abs = coef_df.sort_values("abs_coef", ascending=False)

print("상위 양의 계수 5개:\n", coef_sorted_pos.head(5), "\n")
print("절댓값 기준 상위 5개:\n", coef_sorted_abs.head(5), "\n")

# 가장 긍정/부정적 영향 특성
pos_top = coef_sorted_pos.iloc[0]
neg_top = coef_df.sort_values("coef").iloc[0]

print(f"가장 긍정적 영향: {pos_top['feature']} (coef={pos_top['coef']:.4f})")
print(f"가장 부정적 영향: {neg_top['feature']} (coef={neg_top['coef']:.4f})")

# 4. 시각화
plt.figure(figsize=(11, 5))
order = coef_df.sort_values("coef")
plt.bar(order["feature"], order["coef"])
plt.axhline(0, linewidth=1)
plt.xticks(rotation=45, ha="right")
plt.ylabel("Coefficient")
plt.title("Linear Regression Coefficients (Standardized Features)")
plt.tight_layout()
plt.show()

# 5. 결과 저장 (선택)
coef_df.to_csv("boston_coef.csv", index=False)
print("Saved: boston_coef.csv")
