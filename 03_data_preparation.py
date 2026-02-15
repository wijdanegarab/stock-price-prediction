import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

df = pd.read_csv("data_features.csv")
df['Date'] = pd.to_datetime(df['Date'])


y = df['Target'].values

feature_cols = [col for col in df.columns if col not in ['Date', 'Stock', 'Close', 'Volume', 'Target']]
X = df[feature_cols].values

n_samples = len(X)
split_idx = int(0.8 * n_samples)

X_train = X[:split_idx]
X_test = X[split_idx:]
y_train = y[:split_idx]
y_test = y[split_idx:]



train_dist = np.bincount(y_train)

test_dist = np.bincount(y_test)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

np.save("X_train.npy", X_train_scaled)
np.save("X_test.npy", X_test_scaled)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("feature_names.pkl", "wb") as f:
    pickle.dump(feature_cols, f)






