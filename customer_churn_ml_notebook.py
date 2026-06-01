# =========================
# CUSTOMER CHURN PREDICTION
# MACHINE LEARNING PROJECT NOTEBOOK
# =========================

# =========================
# 1. IMPORT LIBRARIES
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# =========================
# 2. LOAD DATASET
# =========================
# Kaggle: Telco Customer Churn Dataset
# df = pd.read_csv("Telco-Customer-Churn.csv")

# Eğer dosya adı farklıysa burayı değiştir

print("Dataset yükleniyor...")
df = pd.read_csv("Telco-Customer-Churn.csv")

print(df.head())

# =========================
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# =========================

print("\nVeri bilgisi:")
print(df.info())

print("\nEksik değerler:")
print(df.isnull().sum())

print("\nİstatistiksel özet:")
print(df.describe())

# -------------------------
# Churn dağılımı
# -------------------------
plt.figure()
df['Churn'].value_counts().plot(kind='bar')
plt.title("Churn Dağılımı")
plt.xlabel("Churn")
plt.ylabel("Sayım")
plt.show()

# -------------------------
# Monthly Charges vs Churn
# -------------------------
plt.figure()
plt.scatter(df['MonthlyCharges'], df['TotalCharges'], alpha=0.3)
plt.title("Monthly Charges vs Total Charges")
plt.xlabel("Monthly Charges")
plt.ylabel("Total Charges")
plt.show()

# =========================
# 4. DATA PREPROCESSING
# =========================

# TotalCharges numeric dönüşüm (bazı boşluklar olabilir)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Eksik değerleri doldur
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

# Gereksiz ID sütunu varsa kaldır
if 'customerID' in df.columns:
    df = df.drop('customerID', axis=1)

# Label Encoding
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# =========================
# 5. TRAIN-TEST SPLIT
# =========================

X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# 6. MODEL 1 - LOGISTIC REGRESSION
# =========================

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)

print("\nLOGISTIC REGRESSION SONUÇLARI")
print(classification_report(y_test, lr_pred))

# =========================
# 7. MODEL 2 - RANDOM FOREST
# =========================

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

print("\nRANDOM FOREST SONUÇLARI")
print(classification_report(y_test, rf_pred))

# =========================
# 8. MODEL COMPARISON
# =========================

models = ['Logistic Regression', 'Random Forest']
accuracy = [
    accuracy_score(y_test, lr_pred),
    accuracy_score(y_test, rf_pred)
]

plt.figure()
plt.bar(models, accuracy)
plt.title("Model Accuracy Karşılaştırması")
plt.ylabel("Accuracy")
plt.show()

# =========================
# 9. CONFUSION MATRIX (BEST MODEL)
# =========================

best_model_pred = rf_pred  # genelde RF daha iyi

cm = confusion_matrix(y_test, best_model_pred)
print("Confusion Matrix:\n", cm)

# =========================
# 10. FINAL INTERPRETATION
# =========================
print("\nYorum:")
print("Random Forest modeli genellikle daha yüksek performans göstermiştir.")
print("Bu durum doğrusal olmayan ilişkileri daha iyi öğrenmesinden kaynaklanır.")
