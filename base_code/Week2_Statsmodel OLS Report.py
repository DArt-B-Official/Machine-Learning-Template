# Week 2: statsmodels OLS Report (Base Code)
# - 전제: Week 1(저번 과제)에서 만든 설계/전처리 결과 X, y가 이미 준비되어 있음
# - 목표: OLS 적합 → summary 출력 → p-value / R^2 / Adj. R^2 해석

import numpy as np
import pandas as pd
import statsmodels.api as sm


# 0) (선택) week1 산출물을 불러오는 부분
#    만약 X, y가 메모리에 이미 있다면 이 블록은 건너뛰세요.
# X = ...  # Week1에서 만든 feature DataFrame
# y = ...  # Week1에서 만든 target Series/ndarray

# 안전장치: y를 1차원으로 보장
y = np.asarray(y).ravel()

# 1) 상수항 추가 + OLS 적합
X_const = sm.add_constant(X, has_constant='add')  
model   = sm.OLS(y, X_const)
result  = model.fit()   # 기본 OLS (최소제곱)

# (선택) 강건표준오차로도 보고 싶으면:
# result = model.fit(cov_type='HC1')

# 2) 리포트 출력
print(result.summary())

# 3) p-value / 유의미 변수 표 만들기
summary_df = pd.DataFrame({
    "coef": result.params,
    "std_err": result.bse,
    "t": result.tvalues,
    "p_value": result.pvalues
})

# 유의수준 설정 (베이 스 코드에는 0.05로 제공 -> 변경 가능)
alpha = 0.05
summary_df["significant@0.05"] = summary_df["p_value"] < alpha

# 상수항(const) 제외, 유의 변수만 보기
coef_only = summary_df.drop(index="const", errors="ignore")
display(coef_only.sort_values("p_value"))


# 4) R-squared / Adj. R-squared 확인
print(f"R-squared        : {result.rsquared:.4f}")
print(f"Adj. R-squared   : {result.rsquared_adj:.4f}")


# 5) (선택) 해석 템플릿 출력: 노트북에 바로 적기 위한 가이드
print("\n[해석 가이드]")
print("- p-value < 0.05 변수: 통계적으로 유의 → 타깃에 의미있는 관계 가능성")
print("- R-squared: 전체 분산 대비 모델이 설명한 비율")
print("- Adj. R-squared: 변수 수를 고려한 설명력 (과적합 페널티 반영)")
print("- 위 표에서 p-value가 매우 큰 변수는 제거 후보(도메인 고려 필수)")
print("- 모델 설명력/잔차 패턴을 추가 그래프로 점검하면 더 좋음 (예: 잔차 vs 예측)")

# 6) (선택) 결과 저장
