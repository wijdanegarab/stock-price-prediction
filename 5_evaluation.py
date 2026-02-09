import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             roc_curve, auc)
import shap


# ÉTAPE 5: EVALUATION & FEATURE IMPORTANCE


print("=" * 60)
print("EVALUATION & FEATURE IMPORTANCE")
print("=" * 60)

# Load data
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

# Load models
with open("model_logistic_regression.pkl", "rb") as f:
    lr = pickle.load(f)

with open("model_random_forest.pkl", "rb") as f:
    rf = pickle.load(f)

with open("model_xgboost.pkl", "rb") as f:
    xgb_model = pickle.load(f)

# Load feature names
with open("feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

print(f"\nModels loaded:")
print(f"  - Logistic Regression")
print(f"  - Random Forest")
print(f"  - XGBoost")
print(f"\nFeatures: {len(feature_names)}")


# 1. CONFUSION MATRICES


print(f"\n" + "=" * 60)
print("CONFUSION MATRICES")
print("=" * 60)

y_pred_lr = lr.predict(X_test)
y_pred_rf = rf.predict(X_test)
y_pred_xgb = xgb_model.predict(X_test)

cm_lr = confusion_matrix(y_test, y_pred_lr)
cm_rf = confusion_matrix(y_test, y_pred_rf)
cm_xgb = confusion_matrix(y_test, y_pred_xgb)

print(f"\nLogistic Regression:")
print(cm_lr)
print(f"  TN={cm_lr[0,0]}, FP={cm_lr[0,1]}")
print(f"  FN={cm_lr[1,0]}, TP={cm_lr[1,1]}")

print(f"\nRandom Forest:")
print(cm_rf)
print(f"  TN={cm_rf[0,0]}, FP={cm_rf[0,1]}")
print(f"  FN={cm_rf[1,0]}, TP={cm_rf[1,1]}")

print(f"\nXGBoost:")
print(cm_xgb)
print(f"  TN={cm_xgb[0,0]}, FP={cm_xgb[0,1]}")
print(f"  FN={cm_xgb[1,0]}, TP={cm_xgb[1,1]}")


# 2. FEATURE IMPORTANCE - TREE MODELS


print(f"\n" + "=" * 60)
print("FEATURE IMPORTANCE - TREE MODELS")
print("=" * 60)

# Random Forest importance
rf_importance = rf.feature_importances_
rf_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': rf_importance
}).sort_values('Importance', ascending=False)

print(f"\nRandom Forest - Top 10 Features:")
print(rf_importance_df.head(10).to_string(index=False))

# XGBoost importance
xgb_importance = xgb_model.feature_importances_
xgb_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': xgb_importance
}).sort_values('Importance', ascending=False)

print(f"\nXGBoost - Top 10 Features:")
print(xgb_importance_df.head(10).to_string(index=False))

# Save importance
rf_importance_df.to_csv("feature_importance_rf.csv", index=False)
xgb_importance_df.to_csv("feature_importance_xgb.csv", index=False)

print(f"\n✓ Feature importance saved")


# 3. SHAP VALUES (XGBoost)


print(f"\n" + "=" * 60)
print("SHAP VALUES (XGBoost)")
print("=" * 60)

print(f"\nCalculating SHAP values...")

# Create SHAP explainer
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

# Get mean absolute SHAP values (feature importance)
shap_importance = np.abs(shap_values).mean(axis=0)
shap_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'SHAP_Importance': shap_importance
}).sort_values('SHAP_Importance', ascending=False)

print(f"\nSHAP Feature Importance - Top 10:")
print(shap_importance_df.head(10).to_string(index=False))

# Save SHAP importance
shap_importance_df.to_csv("feature_importance_shap.csv", index=False)
print(f"\n✓ SHAP values saved")


# 4. CREATE SUMMARY TABLE


print(f"\n" + "=" * 60)
print("DETAILED METRICS")
print("=" * 60)

models = {
    'Logistic Regression': (lr, y_pred_lr),
    'Random Forest': (rf, y_pred_rf),
    'XGBoost': (xgb_model, y_pred_xgb)
}

metrics_list = []

for model_name, (model, y_pred) in models.items():
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_proba)
    }
    metrics_list.append(metrics)

metrics_df = pd.DataFrame(metrics_list)

print("\n")
print(metrics_df.to_string(index=False))

# Save metrics
metrics_df.to_csv("detailed_metrics.csv", index=False)
print(f"\n✓ Metrics saved: detailed_metrics.csv")


# 5. INSIGHTS & ANALYSIS


print(f"\n" + "=" * 60)
print("KEY INSIGHTS")
print("=" * 60)

best_model_idx = metrics_df['ROC-AUC'].idxmax()
best_model = metrics_df.loc[best_model_idx, 'Model']
best_auc = metrics_df.loc[best_model_idx, 'ROC-AUC']

top_features_rf = rf_importance_df.head(5)['Feature'].tolist()
top_features_xgb = xgb_importance_df.head(5)['Feature'].tolist()
top_features_shap = shap_importance_df.head(5)['Feature'].tolist()

print(f"\n1. Best Performing Model:")
print(f"   {best_model} with ROC-AUC = {best_auc:.4f}")

print(f"\n2. Top 5 Features (Random Forest):")
for i, feat in enumerate(top_features_rf, 1):
    print(f"   {i}. {feat}")

print(f"\n3. Top 5 Features (XGBoost):")
for i, feat in enumerate(top_features_xgb, 1):
    print(f"   {i}. {feat}")

print(f"\n4. Top 5 Features (SHAP):")
for i, feat in enumerate(top_features_shap, 1):
    print(f"   {i}. {feat}")

print(f"\n5. Model Performance:")
print(f"   - Baseline (Logistic): {metrics_df.loc[0, 'Accuracy']:.1%}")
print(f"   - Random Forest: {metrics_df.loc[1, 'Accuracy']:.1%}")
print(f"   - XGBoost: {metrics_df.loc[2, 'Accuracy']:.1%}")
print(f"   - Improvement: +{(metrics_df.loc[2, 'Accuracy'] - metrics_df.loc[0, 'Accuracy'])*100:.1f}%")


# 6. SUMMARY REPORT


print(f"\n" + "=" * 60)
print("SUMMARY REPORT")
print("=" * 60)

summary = f"""
PROJECT SUMMARY
===============

Data:
  - Test samples: {len(y_test)}
  - Features: {len(feature_names)}
  - Classes: 2 (DOWN/UP)

Best Model: {best_model}
  - Accuracy: {metrics_df.loc[best_model_idx, 'Accuracy']:.4f}
  - Precision: {metrics_df.loc[best_model_idx, 'Precision']:.4f}
  - Recall: {metrics_df.loc[best_model_idx, 'Recall']:.4f}
  - F1-Score: {metrics_df.loc[best_model_idx, 'F1-Score']:.4f}
  - ROC-AUC: {best_auc:.4f}

Top 5 Most Important Features:
"""

for i, row in shap_importance_df.head(5).iterrows():
    summary += f"\n  {i+1}. {row['Feature']} ({row['SHAP_Importance']:.4f})"

print(summary)

# Save report
with open("summary_report.txt", "w") as f:
    f.write(summary)
print("\n✓ Summary report saved: summary_report.txt")

print("\n" + "=" * 60)
print("DONE!")
print("=" * 60)
