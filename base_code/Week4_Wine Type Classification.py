# Week 4: Wine Type Classification (Base Code)
# - 목표: Wine Quality(레드/화이트) 데이터로 이진 분류( red=0, white=1 )
# - 모델: DecisionTreeClassifier, 시각화: plot_tree
# - 제출: 트리 그림 + 상위 2개 분기 규칙 설명

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# from sklearn.tree import DecisionTreeClassifier, plot_tree
# import matplotlib.pyplot as plt

# 1. 데이터 불러오기 
# UCI Wine Quality 데이터: CSV 구분자가 ';' 입니다.
# 파일을 같은 폴더에 두었다고 가정합니다. 경로 수정이 필요하면 바꿔서 사용하세요.
PATH_RED   = "winequality-red.csv"
PATH_WHITE = "winequality-white.csv"

red   = pd.read_csv(PATH_RED, sep=';')
white = pd.read_csv(PATH_WHITE, sep=';')

# 1) 레이블 부여: red=0, white=1 
red['target'] = 0
white['target'] = 1

# 2) 합치기 + 섞기 
df = pd.concat([red, white], axis=0, ignore_index=True)
df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

# 3) 특성/타깃 분리 
X = df.drop(columns=['target'])
y = df['target']

# 4) 학습/평가 분할 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

''' 아래는 가이드는 주석으로만 남겨둡니다 '''
# 5) 모델 학습

# 6) 트리 시각화

# 7) 상위 분기 규칙 해석 (힌트도 제공합니다.)
# - 트리의 루트 노드(0번)와 그 다음 레벨(자식 노드들)의 split feature/threshold를 확인하세요.
# 루트의 왼쪽(1번), 오른쪽(2번) 자식 중, 정보이득 큰 쪽을 우선 살펴보세요.
'''
 print("Next split candidates:",
       X.columns[fidx1], "<=", thr1, " | ",
       X.columns[fidx2], "<=", thr2)
'''
#  이 두 분기(루트 + 다음 레벨 한 번)로 "와인 종류를 가르는 상위 2개 규칙"을 서술하세요.

# 8) 성능 확인 (필수는 아닙니다만 궁금하시면 아래의 힌트에 맞게 수행해주세요.)
# from sklearn.metrics import classification_report, confusion_matrix
# y_pred = clf.predict(X_test)
# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred, target_names=['red','white']))
