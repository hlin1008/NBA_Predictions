import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss
import lightgbm as lgb

def run_models(df: pd.DataFrame):
    print("\n--- Starting EC503 Model Pipeline ---")

    # 1. Clean the data for ML
    # Drop rows without our target variable and isolate numeric features
    df = df.dropna(subset=['target_win'])
    
    # Drop IDs and targets to avoid data leakage
    X = df.select_dtypes(include=[np.number]).drop(columns=['target_win', 'game_id', 'team_id_home', 'team_id_away'], errors='ignore')
    y = df['target_win']

    # 2. Chronological Train/Test Split (crucial for sports data)
    # shuffle=False ensures we train on the past and predict the future
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # 3. Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ==========================================
    # EC503 BASELINE: L1 Logistic Regression
    # ==========================================
    print("\nTraining Baseline: L1 Logistic Regression (LASSO)...")
    log_reg = LogisticRegression(penalty='l1', solver='liblinear', random_state=42, max_iter=500)
    
    calibrated_lr = CalibratedClassifierCV(log_reg, cv=5, method='isotonic')
    calibrated_lr.fit(X_train_scaled, y_train)

    lr_preds = calibrated_lr.predict(X_test_scaled)
    lr_probs = calibrated_lr.predict_proba(X_test_scaled)

    print(f"-> Logistic Regression Accuracy: {accuracy_score(y_test, lr_preds) * 100:.2f}%")
    print(f"-> Logistic Regression Log-Loss: {log_loss(y_test, lr_probs):.4f}")

    # ==========================================
    # EC503 ADVANCED: LightGBM
    # ==========================================
    print("\nTraining Advanced: LightGBM Classifier...")
    lgb_model = lgb.LGBMClassifier(random_state=42, n_estimators=100)
    
    calibrated_lgb = CalibratedClassifierCV(lgb_model, cv=5, method='isotonic')
    calibrated_lgb.fit(X_train_scaled, y_train)

    lgb_preds = calibrated_lgb.predict(X_test_scaled)
    lgb_probs = calibrated_lgb.predict_proba(X_test_scaled)

    print(f"-> LightGBM Accuracy: {accuracy_score(y_test, lgb_preds) * 100:.2f}%")
    print(f"-> LightGBM Log-Loss: {log_loss(y_test, lgb_probs):.4f}")

    print("\n--- Top Features (LASSO) ---")
    # Fit the base model explicitly to extract the coefficients
    log_reg.fit(X_train_scaled, y_train)
    coefs = pd.Series(log_reg.coef_[0], index=X.columns)
    print(coefs.abs().sort_values(ascending=False).head(3))

    print("\nPipeline Complete!")