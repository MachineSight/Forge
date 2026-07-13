import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

def repair():
    print("🔄 Reading configuration baseline...")
    # 1. Load config to see the exact feature layout required by the LSTM
    config = joblib.load("config.pkl")
    feature_names = list(config["feature_names"])
    print(f"📊 Target Features: {feature_names}")

    # 2. Read baseline healthy matrix
    df = pd.read_csv("healthy_sample.csv")

    # 3. Align columns exactly how inference.py expects them
    normalized = {str(col).strip().lower(): str(col) for col in df.columns}
    resolved_cols = []
    for feature in feature_names:
        source_name = normalized.get(feature.lower())
        if source_name is None:
            raise ValueError(f"Missing required baseline column: {feature}")
        resolved_cols.append(source_name)

    X_train = df[resolved_cols].copy()
    X_train.columns = feature_names
    X_train = X_train.apply(pd.to_numeric, errors="coerce")
    X_train = X_train.interpolate(limit_direction="both").ffill().bfill()

    # 4. Fit fresh scaler on clean data using modern scikit-learn 1.9.0
    print("⚙️  Fitting fresh memory structure for StandardScaler...")
    scaler = StandardScaler()
    scaler.fit(X_train.values)

    # 5. Overwrite the broken scaler footprint using joblib
    joblib.dump(scaler, "scaler.pkl")
    print("✨ Success! scaler.pkl has been updated for scikit-learn 1.9.0.")

if __name__ == "__main__":
    repair()