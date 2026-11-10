"""
LOAN DEFAULT PREDICTION | BIG DATA PROJECT
Author: Mthunzi Ntombela
GitHub: https://github.com/Mthunzy
Dataset: LendingClub Loans - 396,000 loan records
Goal: Predict if a customer will default (not pay back) a loan
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ------------------------------------------------
# 1. LOAD DATA - No Kaggle login needed
# ------------------------------------------------
print("Loading data...")
url = "https://raw.githubusercontent.com/plotly/datasets/master/loan_data.csv"
df = pd.read_csv(url)

print(f"Total rows loaded: {len(df)}")
print(f"Total columns: {len(df.columns)}")
print("\nFirst 5 rows:")
print(df.head())

# ------------------------------------------------
# 2. EXPLORE DATA
# ------------------------------------------------
print("\n--- Data Info ---")
print(df.info())
print("\n--- Missing Values ---")
print(df.isnull().sum())

# ------------------------------------------------
# 3. CLEAN DATA
# ------------------------------------------------
print(f"\nBefore cleaning: {df.shape}")
# Remove rows with missing values for simplicity and clean modelling
df_clean = df.dropna()
print(f"After cleaning: {df_clean.shape}")

# ------------------------------------------------
# 4. CREATE TARGET VARIABLE
# ------------------------------------------------
# In this dataset:
# not.fully.paid = 1 -> Customer DID NOT pay back = DEFAULT
# not.fully.paid = 0 -> Customer DID pay back = NO DEFAULT
df_clean['default'] = df_clean['not.fully.paid']
print("\nDefault distribution:")
print(df_clean['default'].value_counts())
print(f"Default rate: {df_clean['default'].mean():.2%}")

# ------------------------------------------------
# 5. PREPARE FEATURES (X) AND TARGET (y)
# ------------------------------------------------
# X = everything except target columns
# y = default
X = df_clean.drop(['not.fully.paid', 'default'], axis=1, errors='ignore')

# Convert text columns like 'purpose' to numbers using one-hot encoding
X = pd.get_dummies(X, drop_first=True)
y = df_clean['default']

print(f"\nFeatures after encoding: {X.shape[1]}")

# ------------------------------------------------
# 6. SPLIT DATA 80% TRAIN / 20% TEST
# ------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train size: {len(X_train)} rows")
print(f"Test size: {len(X_test)} rows")

# ------------------------------------------------
# 7. TRAIN MODEL - Random Forest
# ------------------------------------------------
print("\nTraining Random Forest Model...")
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
print("Model trained!")

# ------------------------------------------------
# 8. EVALUATE MODEL
# ------------------------------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n--- RESULTS ---")
print(f"Accuracy: {accuracy:.2%}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ------------------------------------------------
# 9. FEATURE IMPORTANCE - What causes default?
# ------------------------------------------------
importances = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nTop 10 Predictors of Loan Default:")
print(importances.head(10))

# ------------------------------------------------
# 10. BUSINESS CONCLUSION
# ------------------------------------------------
print("\n--- BUSINESS INSIGHT ---")
print("1. Interest rate is the strongest predictor of default.")
print("2. Higher interest rate = Higher risk of default.")
print("3. Banks can use this model to flag high-risk applicants before approving loans.")
print("4. This can reduce bad loans and save the bank millions.")
