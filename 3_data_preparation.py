import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle


# ÉTAPE 3: DATA PREPARATION


print("=" * 60)
print("DATA PREPARATION")
print("=" * 60)

# Load features
df = pd.read_csv("data_features.csv")
df['Date'] = pd.to_datetime(df['Date'])

print(f"\nInput data shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# 1. SEPARATE FEATURES AND TARGET


# Target variable
y = df['Target'].values

# Feature columns (everything except Date, Stock, Close, Volume, Target)
feature_cols = [col for col in df.columns if col not in ['Date', 'Stock', 'Close', 'Volume', 'Target']]
X = df[feature_cols].values

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Target distribution: {np.bincount(y)}")


# 2. SPLIT DATA: TRAIN / TEST


# IMPORTANT: Don't shuffle! Keep time order
# 80% train, 20% test

n_samples = len(X)
split_idx = int(0.8 * n_samples)

X_train = X[:split_idx]
X_test = X[split_idx:]
y_train = y[:split_idx]
y_test = y[split_idx:]

print(f"\n" + "=" * 60)
print("DATA SPLIT")
print("=" * 60)
print(f"\nTrain set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")

print(f"\nTrain target distribution:")
train_dist = np.bincount(y_train)
print(f"  DOWN (0): {train_dist[0]} ({train_dist[0]/len(y_train)*100:.1f}%)")
print(f"  UP   (1): {train_dist[1]} ({train_dist[1]/len(y_train)*100:.1f}%)")

print(f"\nTest target distribution:")
test_dist = np.bincount(y_test)
print(f"  DOWN (0): {test_dist[0]} ({test_dist[0]/len(y_test)*100:.1f}%)")
print(f"  UP   (1): {test_dist[1]} ({test_dist[1]/len(y_test)*100:.1f}%)")


# 3. NORMALIZE FEATURES


print(f"\n" + "=" * 60)
print("FEATURE NORMALIZATION")
print("=" * 60)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nBefore scaling:")
print(f"  X_train mean: {X_train.mean():.4f}, std: {X_train.std():.4f}")
print(f"  X_test mean: {X_test.mean():.4f}, std: {X_test.std():.4f}")

print(f"\nAfter scaling:")
print(f"  X_train mean: {X_train_scaled.mean():.4f}, std: {X_train_scaled.std():.4f}")
print(f"  X_test mean: {X_test_scaled.mean():.4f}, std: {X_test_scaled.std():.4f}")

# 4. SAVE PREPARED DATA


print(f"\n" + "=" * 60)
print("SAVING PREPARED DATA")
print("=" * 60)

# Save as numpy arrays
np.save("X_train.npy", X_train_scaled)
np.save("X_test.npy", X_test_scaled)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

print("✓ Training data saved: X_train.npy, y_train.npy")
print("✓ Test data saved: X_test.npy, y_test.npy")

# Save scaler for future use
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("✓ Scaler saved: scaler.pkl")

# Save feature names
with open("feature_names.pkl", "wb") as f:
    pickle.dump(feature_cols, f)
print("✓ Feature names saved: feature_names.pkl")


# 5. DATA SUMMARY


print(f"\n" + "=" * 60)
print("DATA SUMMARY")
print("=" * 60)

print(f"\nPrepared data:")
print(f"  X_train shape: {X_train_scaled.shape}")
print(f"  X_test shape: {X_test_scaled.shape}")
print(f"  y_train shape: {y_train.shape}")
print(f"  y_test shape: {y_test.shape}")

print(f"\nNumber of features: {len(feature_cols)}")
print(f"\nFeature list:")
for i, col in enumerate(feature_cols, 1):
    print(f"  {i:2d}. {col}")

print("\n" + "=" * 60)
print("DONE!")
print("=" * 60)
