import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import xgboost as xgb
import pickle
import time


print("TRAINING MODELS")


X_train = np.load("X_train.npy")
X_test = np.load("X_test.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

print(f"\nData loaded:")
print(f"  X_train shape: {X_train.shape}")
print(f"  X_test shape: {X_test.shape}")
print(f"  y_train shape: {y_train.shape}")
print(f"  y_test shape: {y_test.shape}")


print("1. LOGISTIC REGRESSION (Baseline)")


start_time = time.time()

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)
y_pred_proba_lr = lr.predict_proba(X_test)[:, 1]

train_time_lr = time.time() - start_time

acc_lr = accuracy_score(y_test, y_pred_lr)
prec_lr = precision_score(y_test, y_pred_lr)
rec_lr = recall_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr)
auc_lr = roc_auc_score(y_test, y_pred_proba_lr)

print(f"\nResults:")
print(f"  Accuracy:  {acc_lr:.4f}")
print(f"  Precision: {prec_lr:.4f}")
print(f"  Recall:    {rec_lr:.4f}")
print(f"  F1-Score:  {f1_lr:.4f}")
print(f"  ROC-AUC:   {auc_lr:.4f}")
print(f"  Train time: {train_time_lr:.2f}s")


with open("model_logistic_regression.pkl", "wb") as f:
    pickle.dump(lr, f)
print("✓ Model saved: model_logistic_regression.pkl")






print("2. RANDOM FOREST")


start_time = time.time()

rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
y_pred_proba_rf = rf.predict_proba(X_test)[:, 1]

train_time_rf = time.time() - start_time


acc_rf = accuracy_score(y_test, y_pred_rf)
prec_rf = precision_score(y_test, y_pred_rf)
rec_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)
auc_rf = roc_auc_score(y_test, y_pred_proba_rf)

print(f"\nResults:")
print(f"  Accuracy:  {acc_rf:.4f}")
print(f"  Precision: {prec_rf:.4f}")
print(f"  Recall:    {rec_rf:.4f}")
print(f"  F1-Score:  {f1_rf:.4f}")
print(f"  ROC-AUC:   {auc_rf:.4f}")
print(f"  Train time: {train_time_rf:.2f}s")


with open("model_random_forest.pkl", "wb") as f:
    pickle.dump(rf, f)
print("saved")


print("3. XGBOOST")


start_time = time.time()

xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1,
    verbose=0
)
xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)
y_pred_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]

train_time_xgb = time.time() - start_time


acc_xgb = accuracy_score(y_test, y_pred_xgb)
prec_xgb = precision_score(y_test, y_pred_xgb)
rec_xgb = recall_score(y_test, y_pred_xgb)
f1_xgb = f1_score(y_test, y_pred_xgb)
auc_xgb = roc_auc_score(y_test, y_pred_proba_xgb)

print(f"\nResults:")
print(f"  Accuracy:  {acc_xgb:.4f}")
print(f"  Precision: {prec_xgb:.4f}")
print(f"  Recall:    {rec_xgb:.4f}")
print(f"  F1-Score:  {f1_xgb:.4f}")
print(f"  ROC-AUC:   {auc_xgb:.4f}")
print(f"  Train time: {train_time_xgb:.2f}s")


with open("model_xgboost.pkl", "wb") as f:
    pickle.dump(xgb_model, f)
print("saved")




print("MODEL COMPARISON")


comparison = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest', 'XGBoost'],
    'Accuracy': [acc_lr, acc_rf, acc_xgb],
    'Precision': [prec_lr, prec_rf, prec_xgb],
    'Recall': [rec_lr, rec_rf, rec_xgb],
    'F1-Score': [f1_lr, f1_rf, f1_xgb],
    'ROC-AUC': [auc_lr, auc_rf, auc_xgb],
    'Train Time (s)': [train_time_lr, train_time_rf, train_time_xgb]
})


print(comparison.to_string(index=False))


comparison.to_csv("model_comparison.csv", index=False)



print("BEST MODEL")


best_model_idx = comparison['ROC-AUC'].idxmax()
best_model_name = comparison.loc[best_model_idx, 'Model']
best_model_auc = comparison.loc[best_model_idx, 'ROC-AUC']

print(f"\n Best Model: {best_model_name}")
print(f"  ROC-AUC: {best_model_auc:.4f}")


print("SAVING PREDICTIONS")


predictions_df = pd.DataFrame({
    'y_true': y_test,
    'y_pred_lr': y_pred_lr,
    'y_proba_lr': y_pred_proba_lr,
    'y_pred_rf': y_pred_rf,
    'y_proba_rf': y_pred_proba_rf,
    'y_pred_xgb': y_pred_xgb,
    'y_proba_xgb': y_pred_proba_xgb
})

predictions_df.to_csv("predictions.csv", index=False)



