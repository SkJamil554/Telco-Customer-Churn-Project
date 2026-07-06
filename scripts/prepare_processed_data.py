import os, sys
import pandas as pd

# make src importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.preprocess import preprocess_data
from src.features.build_features import build_features

RAW = "data/raw/Telco-Customer-Churn.csv"
OUT = "data/processed/telco_churn_processed.csv"

def main(raw_path: str = RAW, out_path: str = OUT):
    # 1) load raw
    df = pd.read_csv(raw_path)

    # 2) preprocess (drops id, fixes TotalCharges, etc.)
    df = preprocess_data(df, target_col="Churn")

    # 3) ensure target is 0/1 only if still object
    if "Churn" in df.columns and df["Churn"].dtype == "object":
        df["Churn"] = df["Churn"].str.strip().map({"No": 0, "Yes": 1}).astype("Int64")

    # sanity checks
    assert df["Churn"].isna().sum() == 0, "Churn has NaNs after preprocess"
    assert set(df["Churn"].unique()) <= {0, 1}, "Churn not 0/1 after preprocess"

    # 4) features
    df_processed = build_features(df, target_col="Churn")

    # 5) save
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_processed.to_csv(out_path, index=False)
    print(f"✅ Processed dataset saved to {out_path} | Shape: {df_processed.shape}")


if __name__ == "__main__":
    main()
