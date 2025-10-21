# Week 5: RandomForest 튜닝 비교 (Baseline Code)
# - 목표: 3주차 와인 품질(레드/화이트) 데이터 재사용, RandomForest 튜닝 비교
# - 실험: n_estimators ∈ {100, 2000}, max_features ∈ {"sqrt","log2", None, 0.5} (총 8조합)
# - 지표: Accuracy, F1-macro (+ 학습시간), 최고 조합의 feature_importances_ 해석
# - 제출: (1) 8조합 결과표, (2) 최고 조합 선택 및 근거, (3) 중요 변수 Top-10

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, make_scorer

# =============================
# 1) 데이터 불러오기 (3주차와 동일)
# =============================
PATH_RED   = "winequality-red.csv"    # UCI 데이터, 구분자 ';'
PATH_WHITE = "winequality-white.csv"

red   = pd.read_csv(PATH_RED, sep=';')
white = pd.read_csv(PATH_WHITE, sep=';')

# 레이블: red=0, white=1
red['target'] = 0
white['target'] = 1

# 합치고 섞기
df = pd.concat([red, white], axis=0, ignore_index=True)
df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

# 특성/타깃
X = df.drop(columns=['target'])
y = df['target']

# 학습/평가 분할 (동일 분할 유지)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# =============================
# 2) 실험 준비
# =============================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 다중분류 AUC가 필요하면 OVR, 이진이면 기본 사용 가능
scorers = {
    'acc': make_scorer(accuracy_score),
    'f1m': make_scorer(f1_score, average='macro'),
}
try:
    scorers['auc'] = make_scorer(roc_auc_score, needs_proba=True, multi_class='ovr')
except Exception:
    pass

grid_ne = [100, 2000]
grid_mf = ['sqrt', 'log2', None, 0.5]

results = []
feat_imp_by_setting = {}  # 조합별 평균 feature_importances_ 저장

# =============================
# 3) 8개 조합 교차검증
# =============================
for ne in grid_ne:
    for mf in grid_mf:
        rf = RandomForestClassifier(
            n_estimators=ne,
            max_features=mf,
            random_state=42,
            n_jobs=-1
        )
        t0 = time.time()
        out = cross_validate(
            rf, X_train, y_train,
            scoring=scorers, cv=cv, n_jobs=-1,
            return_estimator=True
        )
        fit_time = out['fit_time'].mean()
        scores = {m: out[f'test_{m}'].mean() for m in scorers}

        # 각 fold의 feature_importances_ 평균
        importances = np.mean([est.feature_importances_ for est in out['estimator']], axis=0)
        feat_imp_by_setting[(ne, mf)] = importances

        results.append({
            'n_estimators': ne,
            'max_features': mf,
            'acc': scores.get('acc', np.nan),
            'f1m': scores.get('f1m', np.nan),
            'auc': scores.get('auc', np.nan),
            'cv_fit_time_sec': fit_time,
            'cv_score_time_sec': out['score_time'].mean(),
            'cv_runtime_sec': time.time() - t0
        })

# 결과표 정리 (성능 우선 정렬)
df_res = pd.DataFrame(results).sort_values(['f1m','acc'], ascending=False).reset_index(drop=True)
print("\n=== 8개 조합 결과표 (교차검증 평균) ===")
print(df_res)

# =============================
# 4) 최고 조합 선택 + Test 성능 확인
# =============================
best_row = df_res.iloc[0]
best_ne = int(best_row['n_estimators'])
best_mf = best_row['max_features']

print(f"\n[선정] best setting -> n_estimators={best_ne}, max_features={best_mf}")

best_model = RandomForestClassifier(
    n_estimators=best_ne, max_features=best_mf,
    random_state=42, n_jobs=-1
).fit(X_train, y_train)

y_pred = best_model.predict(X_test)
acc_test = accuracy_score(y_test, y_pred)
f1m_test = f1_score(y_test, y_pred, average='macro')
print(f"[Test] Accuracy={acc_test:.4f}, F1-macro={f1m_test:.4f}")

# =============================
# 5) 중요 변수 해석 (Top-10)
# =============================
# 교차검증 평균 importances 사용(일반화된 해석을 위해)
best_imp = feat_imp_by_setting[(best_ne, best_mf)]
feat_names = X.columns.to_list()

top_k = 10
idx_sorted = np.argsort(best_imp)[::-1][:top_k]
print("\n=== 중요 변수 Top-10 (교차검증 평균) ===")
for rank, i in enumerate(idx_sorted, 1):
    print(f"{rank:>2}. {feat_names[i]}: {best_imp[i]:.4f}")

# (선택) 막대 그래프를 그리고 싶다면 아래 주석 해제
# import matplotlib.pyplot as plt
# plt.figure(figsize=(8,4))
# plt.bar([feat_names[i] for i in idx_sorted], best_imp[idx_sorted])
# plt.xticks(rotation=45, ha='right')
# plt.title(f"Feature Importances (best: ne={best_ne}, mf={best_mf})")
# plt.tight_layout(); plt.show()

# =============================
# 6) 리포트에 포함할 최소 산출물 가이드 (주석)
# =============================
# - 표1: df_res (8조합 Acc/F1m/AUC/시간) 캡처 또는 표 붙이기
# - 문단: best 조합 선정 근거(성능/시간 트레이드오프 포함)
# - 표/리스트: 중요 변수 Top-10 + 간단 해석(왜 중요할지 도메인 추론 2~3줄)
# - 부록(선택): Test 성능 수치, 혼동행렬/분류리포트
