from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# 데이터 불러오기
iris = load_iris()

X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='species')

print("target 값 종류 :", np.unique(y))

# EDA 시각화
sns.pairplot(pd.concat([X, y], axis=1), hue='species')
plt.savefig("species.png")

# 학습용/검증용 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 랜덤 포레스트
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# 랜덤 포레스트 결과 출력
print("\n 랜덤 포레스트 결과 출력")
print(classification_report(y_test, rf_pred))

# 로지스틱 회귀
lg_model = LogisticRegression(max_iter=200)
lg_model.fit(X_train, y_train)
lg_pred = lg_model.predict(X_test)

# 로지스틱 회귀 결과 출력
print("\n 로지스틱 회귀 결과 출력")
print(classification_report(y_test, lg_pred))

# Confusion Matrix 출력
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, rf_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Random Forest Confusion Matrix")
#plt.title("Logistic Confusion Matrix")
#plt.savefig("lg_model_matrix")
plt.savefig("rf_model_matrix")