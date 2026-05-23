# ==========================================
# HEART DISEASE PREDICTION USING ML
# ==========================================

# Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv(r"C:\Users\indhu\Downloads\archive (2)\heart.csv")

print("\n========== FIRST 5 ROWS ==========\n")
print(df.head())

print("\n========== DATASET INFO ==========\n")
print(df.info())

print("\n========== DATASET SHAPE ==========\n")
print(df.shape)

print("\n========== STATISTICS ==========\n")
print(df.describe())

# ==========================================
# DATA CLEANING
# ==========================================

print("\n========== MISSING VALUES ==========\n")
print(df.isnull().sum())

# Remove duplicates

df = df.drop_duplicates()

print("\nDataset shape after removing duplicates:")
print(df.shape)

# ==========================================
# EXPLORATORY DATA ANALYSIS
# ==========================================

# Heart disease count

plt.figure(figsize=(6,5))

sns.countplot(
    x='target',
    data=df
)

plt.title("Heart Disease Count")

plt.xlabel(
    "Target (0 = No Disease, 1 = Disease)"
)

plt.ylabel("Count")

plt.show()

# ------------------------------------------

# Age distribution

plt.figure(figsize=(8,5))

sns.histplot(
    df['age'],
    bins=20,
    kde=True
)

plt.title("Age Distribution")

plt.xlabel("Age")

plt.show()

# ------------------------------------------

# Gender vs Disease

plt.figure(figsize=(7,5))

sns.countplot(
    x='sex',
    hue='target',
    data=df
)

plt.title(
    "Gender vs Heart Disease"
)

plt.xlabel(
    "Sex (0=Female, 1=Male)"
)

plt.show()

# ------------------------------------------

# Correlation Heatmap

plt.figure(
    figsize=(12,8)
)

sns.heatmap(
    df.corr(),
    annot=True,
    cmap='coolwarm'
)

plt.title(
    "Correlation Heatmap"
)

plt.show()

# ==========================================
# FEATURE SELECTION
# ==========================================

X = df.drop(
    'target',
    axis=1
)

y = df['target']

print("\nFeatures Shape:", X.shape)

print("Target Shape:", y.shape)

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Shape:")
print(X_train.shape)

print("\nTesting Shape:")
print(X_test.shape)

# ==========================================
# MODEL TRAINING
# ==========================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

print("\nModel Training Completed")

# ==========================================
# PREDICTION
# ==========================================

y_pred = model.predict(
    X_test
)

print("\nPredictions:\n")
print(y_pred)

# ==========================================
# MODEL EVALUATION
# ==========================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nAccuracy Score:")
print(accuracy)

# Confusion Matrix

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:\n")
print(cm)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    cmap='Blues'
)

plt.title(
    "Confusion Matrix"
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.show()

# Classification Report

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred
    )
)

# ==========================================
# FEATURE IMPORTANCE
# ==========================================

importance = pd.DataFrame({

    'Feature':
    X.columns,

    'Importance':
    model.feature_importances_

})

importance = importance.sort_values(
    by='Importance',
    ascending=False
)

print(
    "\nFeature Importance:\n"
)

print(
    importance
)

plt.figure(
    figsize=(10,6)
)

sns.barplot(
    x='Importance',
    y='Feature',
    data=importance
)

plt.title(
    "Feature Importance"
)

plt.show()

# ==========================================
# SAMPLE PATIENT PREDICTION
# ==========================================

sample_patient = np.array([[
    52,
    1,
    0,
    125,
    212,
    0,
    1,
    168,
    0,
    1.0,
    2,
    2,
    3
]])

prediction = model.predict(
    sample_patient
)

print(
    "\nPatient Prediction:"
)

if prediction[0] == 1:

    print(
        "Heart Disease Detected"
    )

else:

    print(
        "No Heart Disease"
    )

# ==========================================
# END
# ==========================================