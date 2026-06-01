# ==============================
# TITANIC ML FINAL PROJECT
# TEK DOSYA - KOPYALA & ÇALIŞTIR
# ==============================

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ==============================
# 1. VERİYİ YÜKLE
# ==============================
df = pd.read_csv("train.csv")

print("\n--- İLK 5 SATIR ---")
print(df.head())

print("\n--- EKSİK VERİLER ---")
print(df.isnull().sum())

# ==============================
# 2. VERİ TEMİZLEME
# ==============================
df["Age"].fillna(df["Age"].median(), inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)
df.drop(columns=["Cabin"], inplace=True)

# ==============================
# 3. EDA (GÖRSELLEŞTİRME)
# ==============================
plt.figure()
sns.countplot(x="Survived", data=df)
plt.title("Hayatta Kalma Dağılımı")
plt.show()

plt.figure()
sns.countplot(x="Survived", hue="Sex", data=df)
plt.title("Cinsiyete Göre Hayatta Kalma")
plt.show()

plt.figure()
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.title("Korelasyon Matrisi")
plt.show()

# ==============================
# 4. LABEL ENCODING
# ==============================
le = LabelEncoder()

df["Sex"] = le.fit_transform(df["Sex"])
df["Embarked"] = le.fit_transform(df["Embarked"])

# ==============================
# 5. VERİ SETİ HAZIRLAMA
# ==============================
X = df.drop("Survived", axis=1)
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# 6. MODELLER
# ==============================
log_model = LogisticRegression(max_iter=1000)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

log_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# ==============================
# 7. TAHMİN
# ==============================
log_pred = log_model.predict(X_test)
rf_pred = rf_model.predict(X_test)

# ==============================
# 8. SONUÇLAR
# ==============================
print("\n==============================")
print("LOGISTIC REGRESSION")
print("==============================")
print("Accuracy:", accuracy_score(y_test, log_pred))
print(classification_report(y_test, log_pred))

print("\n==============================")
print("RANDOM FOREST")
print("==============================")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print(classification_report(y_test, rf_pred))

print("\nPROJE TAMAMLANDI")
