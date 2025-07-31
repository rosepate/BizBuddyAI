import pandas as pd

def load_data():
    sheet_url = "https://docs.google.com/spreadsheets/d/1N0cGSLkm9k-p5aIDsPbVp-fOl5SZVitJfQLs9LThANk/export?format=csv"
    df = pd.read_csv(sheet_url)
    if "Sale Date" in df.columns:
        df.rename(columns={"Sale Date": "Date"}, inplace=True)
    return df

df = load_data()

def detect_z_score_anomalies(df: pd.DataFrame, column: str, threshold: float = 3.0) -> pd.DataFrame:
    """
    Returns a DataFrame with z-scores and an 'Anomaly' column (True/False) for the specified column.
    """
    df = df.copy()
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")
    mean = df[column].mean()
    std = df[column].std()
    df['z_score'] = (df[column] - mean) / std
    df['Anomaly'] = df['z_score'].abs() > threshold
    return df

if __name__ == "__main__":
    # Example usage for Inventory_After
    sheet_url = "https://docs.google.com/spreadsheets/d/1N0cGSLkm9k-p5aIDsPbVp-fOl5SZVitJfQLs9LThANk/export?format=csv"
    df = load_data()
    print(f"✅ Loaded {len(df)} rows.")

    for col in ['Units_Sold', 'Inventory_After']:
        if col in df.columns:
            anomalies = detect_z_score_anomalies(df, column=col, threshold=3)
            detected = anomalies[anomalies['Anomaly']]
            print(f"\nAnomalies in {col}:")
            if detected.empty:
                print("✅ No anomalies detected.")
            else:
                print(detected)
        else:
            print(f"⚠️ '{col}' column not found. Available columns: {df.columns.tolist()}")