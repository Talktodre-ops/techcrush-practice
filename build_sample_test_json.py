"""Build sample_test_rows.json from insurance_claims.csv using the same pipeline as the notebook."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
# Prefer cleaned CSV so ? -> UNKNOWN etc. matches the preprocessor fit on df1 from cleaned data
CSV_PATH = HERE / "cleaned_insurance_claims.csv"
if not CSV_PATH.exists():
    CSV_PATH = HERE / "insurance_claims.csv"


def add_features(d: pd.DataFrame) -> pd.DataFrame:
    out = d.copy()
    out["policy_bind_date"] = pd.to_datetime(out["policy_bind_date"], errors="coerce")
    out["incident_date"] = pd.to_datetime(out["incident_date"], errors="coerce")
    out["incident_dow"] = out["incident_date"].dt.dayofweek
    out["is_weekend"] = out["incident_dow"].isin([5, 6]).astype(int)
    out["days_policy_to_incident"] = (
        out["incident_date"] - out["policy_bind_date"]
    ).dt.days
    out["premium_per_month"] = np.where(
        out["months_as_customer"] > 0,
        out["policy_annual_premium"] / out["months_as_customer"],
        np.nan,
    )
    out["claim_nonzero_injury_share"] = np.where(
        out["total_claim_amount"] > 0,
        out["injury_claim"] / out["total_claim_amount"],
        np.nan,
    )
    return out


def row_to_jsonable(row: pd.Series) -> dict:
    out = {}
    for k, v in row.items():
        if pd.isna(v):
            out[k] = None
        elif isinstance(v, (np.integer, np.int64, np.int32)):
            out[k] = int(v)
        elif isinstance(v, (np.floating, np.float64, np.float32)):
            out[k] = float(v)
        else:
            out[k] = str(v)
    return out


def main():
    # Cleaned file: no need to map ? (already UNKNOWN / consistent strings)
    df = pd.read_csv(CSV_PATH, parse_dates=["policy_bind_date", "incident_date"])
    if "_c39" in df.columns:
        df = df.drop(columns=["_c39"])

    df2 = add_features(df)
    df2["target"] = (
        df2["fraud_reported"].astype(str).str.upper().str.strip() == "Y"
    ).astype(int)

    DROP = [
        "policy_number",
        "policy_bind_date",
        "incident_date",
        "incident_location",
        "fraud_reported",
    ]
    df2 = df2.drop(columns=[c for c in DROP if c in df2.columns])

    feature_cols = [c for c in df2.columns if c != "target"]
    X = df2[feature_cols]

    samples = []
    for i in range(min(3, len(df2))):
        row = X.iloc[i]
        samples.append(
            {
                "meta": {
                    "csv_row_index": i,
                    "source": "insurance_claims.csv data rows 2-4 (0-based data rows after header)",
                    "target_fraud_1_if_yes": int(df2["target"].iloc[i]),
                },
                "features": row_to_jsonable(row),
            }
        )

    payload = {
        "description": (
            "Feature rows matching X_train_df after add_features + DROP. "
            "Load with pd.DataFrame([features])[columns] then preprocess.transform."
        ),
        "feature_column_order": feature_cols,
        "samples": samples,
    }

    out_path = HERE / "sample_test_rows.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    cols_path = HERE / "fraud_feature_columns.json"
    cols_path.write_text(json.dumps(feature_cols, indent=2), encoding="utf-8")

    print("Wrote:", out_path)
    print("Wrote:", cols_path)
    print("Columns:", len(feature_cols))


if __name__ == "__main__":
    main()
