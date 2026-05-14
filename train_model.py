import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


# =========================
# LOAD DATASET
# =========================
df = pd.read_csv("dataset/Synthetic_dataset.csv")

print("Dataset Shape:", df.shape)


# =========================
# DATA CLEANING
# =========================
df = df[df['age'] >= 20]
df = df[df['crp'] <= 20]

df['eosinophil_pct'] = df['eosinophil_pct'].clip(lower=0)

df = df.dropna(subset=['Albumin', 'Glucose', 'Creatinine'])

df = df.replace([np.inf, -np.inf], np.nan)

numeric_cols = df.select_dtypes(include=np.number).columns
df[numeric_cols] = df[numeric_cols].apply(lambda x: x.fillna(x.median()))


# =========================
# FEATURES & TARGET
# =========================
selected_features = [
    'gender', 'wbc',
    'neutrophil_pct', 'lymphocyte_pct', 'monocyte_pct',
    'hemoglobin', 'rdw', 'platelets',
    'crp', 'Albumin', 'Glucose', 'Creatinine',
    'Uric_Acid', 'Total_Cholesterol', 'BUN_Blood_Urea_Nitrogen',
    'nlr', 'il6', 'tnf_alpha',
    'cd4_tcell_count', 'cd8_tcell_count', 'cd4_cd8_ratio',
    'bmi', 'smoking_status'
]

X = df[selected_features].copy()

# Automatically convert strings (Gender, Smoking) to numeric codes
for col in X.select_dtypes(include=['object']).columns:
    X[col] = X[col].astype('category').cat.codes

y = df['age']

# =========================
# TRAIN-TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# =========================
# MODEL
# =========================
model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    enable_categorical=True
)

model.fit(X_train, y_train)


# =========================
# PREDICTION
# =========================
y_pred = model.predict(X_test)


# =========================
# METRICS
# =========================
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nMODEL PERFORMANCE")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.2f}")


# =========================
# IMMUNE AGE FRAME
# =========================
test_results = X_test.copy()
test_results["chronological_age"] = y_test.values
test_results["immune_age"] = y_pred
test_results["aging_gap"] = test_results["immune_age"] - test_results["chronological_age"]


# =========================
# CLASSIFICATION
# =========================
def classify_gap(x):
    if abs(x) <= 3:
        return "Physiological Aging"
    elif x > 3:
        return "Accelerated Aging"
    else:
        return "Slow Aging"

test_results["aging_type"] = test_results["aging_gap"].apply(classify_gap)


# =========================
# SHAP (SAFE VERSION)
# =========================
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)

# OPTIONAL VISUALS (commented for VS Code safety)
# shap.summary_plot(shap_values, X_test)
# shap.plots.bar(shap_values)


# =========================
# FEATURE IMPORTANCE
# =========================
shap_importance = np.abs(shap_values.values).mean(axis=0)

feature_importance = pd.DataFrame({
    "feature": X_test.columns,
    "importance": shap_importance
}).sort_values(by="importance", ascending=False)

print("\nTOP 10 FEATURES")
print(feature_importance.head(10))


# =========================
# CLINICAL INSIGHTS
# =========================
print("\nCLINICAL INSIGHTS")

top_features = feature_importance.head(5)

for _, row in top_features.iterrows():
    f = row["feature"].lower()

    print(f"\n{row['feature']}:")

    if "crp" in f:
        print("→ Inflammation marker affecting immune aging.")
    elif "creatinine" in f:
        print("→ Kidney function linked to biological aging.")
    elif "glucose" in f:
        print("→ Metabolic stress indicator.")
    elif "albumin" in f:
        print("→ Nutrition and protein status marker.")
    else:
        print("→ Major contributor to immune age.")


# =========================
# SAVE MODEL
# =========================
joblib.dump(model, "model.pkl")
joblib.dump(X.columns, "X_columns.pkl")
joblib.dump(explainer, "explainer.pkl")

print("\nMODEL & COLUMNS SAVED SUCCESSFULLY")
