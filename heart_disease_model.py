# -*- coding: utf-8 -*-
"""heart_disease_model.ipynb
    https://colab.research.google.com/drive/1sCgOoKoUfUamIj5BDGdqFyKsA2MVFnMG
"""

#1 ===== INSTALL / IMPORT =====
# !pip install -q seaborn   
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#2 STABLE CSV (Kaggle cleaned version)
df = pd.read_csv('https://storage.googleapis.com/download.tensorflow.org/data/heart.csv')
df.head()

#3 5-LINE QUICK EXPLORATION
print("Shape:", df.shape)
print("Missing:", df.isna().sum().sum())
print("Duplicates:", df.duplicated().sum())
print("Target counts:\n", df['target'].value_counts())

#4 MINI-CLEAN: drop the single duplicate
df = df.drop_duplicates()
print("New shape:", df.shape)

#5 SPLIT 75-25, stratified
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)
print("Train:", X_train.shape, "Test:", X_test.shape)

#6 QUICK ENCODE THEN SPLIT
df = pd.get_dummies(df, columns=['thal'], drop_first=True)   # turns strings → 0/1
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)
print("Train:", X_train.shape, "Test:", X_test.shape)

#7 RETRAIN MODEL AFTER ENCODING
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print("Model re-trained with correct features!")

# PREDICT + METRICS
pred = model.predict(X_test)
acc  = accuracy_score(y_test, pred)
cm   = confusion_matrix(y_test, pred)

print("Accuracy:", round(acc, 2))
print(classification_report(y_test, pred))

# CONFUSION-MATRIX HEATMAP
plt.figure(figsize=(3,2))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Confusion Matrix")
plt.ylabel("Actual"); plt.xlabel("Predicted")
plt.savefig("confusion_matrix.png", dpi=120, bbox_inches='tight')
plt.show()

# 7  PREDICT + METRICS
pred = model.predict(X_test)
acc  = accuracy_score(y_test, pred)
cm   = confusion_matrix(y_test, pred)
print("Accuracy:", round(acc, 2))
print(classification_report(y_test, pred))

# 8  CONFUSION-MATRIX HEATMAP
plt.figure(figsize=(3,2))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Confusion Matrix")
plt.ylabel("Actual"); plt.xlabel("Predicted")
plt.savefig("confusion_matrix.png", dpi=120, bbox_inches='tight')
plt.show()