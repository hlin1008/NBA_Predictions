import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

# 1 & 2. Load the Data & Split Features
# EMERGENCY BYPASS: Generating dummy data to test the pipeline for the report
print("Generating dummy dataset for preliminary pipeline test...")
np.random.seed(42)
# Creating 1000 fake NBA games with 10 random statistical features
X = pd.DataFrame(np.random.rand(1000, 10), columns=[f'Stat_{i}' for i in range(10)])
y = np.random.randint(0, 2, 1000) # Binary Win(1) or Loss(0)

# 3. Train/Test Split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize and Train a Basic LightGBM Classifier
model = lgb.LGBMClassifier(random_state=42)
model.fit(X_train, y_train)

# 5. Predict and Evaluate
y_pred_proba = model.predict_proba(X_test)
y_pred = model.predict(X_test)

# Calculate metrics
acc = accuracy_score(y_test, y_pred)
ll = log_loss(y_test, y_pred_proba)

print(f"Preliminary Accuracy: {acc:.4f}")
print(f"Preliminary Log-Loss: {ll:.4f}")