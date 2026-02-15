import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             roc_curve, auc)
import shap

X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")


with open("model_logistic_regression.pkl", "rb") as f:
    lr = pickle.load(f)

with open("model_random_forest.pkl", "rb") as f:
    rf = pickle.load(f)

with open("model_xgboost.pkl", "rb") as f:
    xgb_model = pickle.load(f)


with open("feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

y_pred_lr = lr.predict(X_test)
y_pred_rf = rf.predict(X_test)
y_pred_xgb = xgb_model.predict(X_test)

cm_lr = confusion_matrix(y_test, y_pred_lr)
cm_rf = confusion_matrix(y_test, y_pred_rf)
cm_xgb = confusion_matrix(y_test, y_pred_xgb)


rf_importance = rf.feature_importances_
rf_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': rf_importance
}).sort_values('Importance', ascending=False)

xgb_importance = xgb_model.feature_importances_
xgb_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': xgb_importance
}).sort_values('Importance', ascending=False)

rf_importance_df.to_csv("feature_importance_rf.csv", index=False)
xgb_importance_df.to_csv("feature_importance_xgb.csv", index=False)


explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)


shap_importance = np.abs(shap_values).mean(axis=0)
shap_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'SHAP_Importance': shap_importance
}).sort_values('SHAP_Importance', ascending=False)

shap_importance_df.to_csv("feature_importance_shap.csv", index=False)

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

metrics_df.to_csv("detailed_metrics.csv", index=False)

best_model_idx = metrics_df['ROC-AUC'].idxmax()
best_model = metrics_df.loc[best_model_idx, 'Model']
best_auc = metrics_df.loc[best_model_idx, 'ROC-AUC']

top_features_rf = rf_importance_df.head(5)['Feature'].tolist()
top_features_xgb = xgb_importance_df.head(5)['Feature'].tolist()
top_features_shap = shap_importance_df.head(5)['Feature'].tolist()

for i, row in shap_importance_df.head(5).iterrows():
    summary += f"\n  {i+1}. {row['Feature']} ({row['SHAP_Importance']:.4f})"

with open("summary_report.txt", "w") as f:
    f.write(summary)



