#ML based Auto Reorder Suggestion
# This module provides functionality to load sales data, create features, train a machine learning model,
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def load_data():
    """
    Loads sales/order data from Google Sheet.
    Assumes columns: 'Date' (order date), 'Receive' (arrival date), 'Product', 'Location', 'Units_Sold', 'Inventory_After'
    """
    sheet_url = "https://docs.google.com/spreadsheets/d/1N0cGSLkm9k-p5aIDsPbVp-fOl5SZVitJfQLs9LThANk/export?format=csv"
    df = pd.read_csv(sheet_url)
    if "Sale Date" in df.columns:
        df.rename(columns={"Sale Date": "Date"}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    if 'Receive' in df.columns:
        df['Receive'] = pd.to_datetime(df['Arrival Date'], errors='coerce')
        df['Lead_Time'] = (df['Receive'] - df['Date']).dt.days
        df['Lead_Time'] = df['Lead_Time'].apply(lambda x: x if x >= 0 else np.nan)
        df['Lead_Time'].fillna(df['Lead_Time'].median(), inplace=True)
    else:
        df['Lead_Time'] = 5  # fallback if Receive column missing
    return df

def create_features(df):
    df = df.sort_values(['Product', 'Location', 'Date'])
    df['Units_Sold_7d'] = df.groupby(['Product', 'Location'])['Units_Sold'].transform(lambda x: x.rolling(7, min_periods=1).sum())
    df['Inventory_After_7d'] = df.groupby(['Product', 'Location'])['Inventory_After'].shift(-7)
    df['Month'] = df['Date'].dt.month
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    threshold = 40
    df['Reorder_Needed'] = (df['Inventory_After_7d'] < threshold).astype(int)
    print("Label distribution:\n", df['Reorder_Needed'].value_counts())
    df = df.dropna(subset=['Inventory_After_7d', 'Lead_Time'])
    return df

from sklearn.model_selection import GridSearchCV

def train_reorder_model(df):
    features = ['Units_Sold_7d', 'Inventory_After', 'Lead_Time', 'Month', 'WeekOfYear', 'DayOfWeek']
    X = df[features]
    y = df['Reorder_Needed']
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
    )
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5, 10]
    }
    grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='f1', n_jobs=-1)
    grid.fit(X_train, y_train)
    print("Best params:", grid.best_params_)
    y_pred = grid.predict(X_test)
    print(classification_report(y_test, y_pred))
    return grid.best_estimator_

def suggest_reorder(df, clf, product, location):
    latest = df[(df['Product'] == product) & (df['Location'] == location)].sort_values('Date').iloc[-1]
    features = pd.DataFrame([{
    'Units_Sold_7d': latest['Units_Sold_7d'],
    'Inventory_After': latest['Inventory_After'],
    'Lead_Time': latest['Lead_Time'],
    'Month': latest['Month'],
    'WeekOfYear': latest['WeekOfYear'],
    'DayOfWeek': latest['DayOfWeek']
    }])
    pred = clf.predict(features)[0]
    if pred == 1:
        return f"🔔 Reorder suggested for {product} at {location} (ML prediction)"
    else:
        return f"✅ No reorder needed for {product} at {location} (ML prediction)"

# Add this at the end of your script for batch suggestions
if __name__ == "__main__":
    df = load_data()
    df = create_features(df)
    clf = train_reorder_model(df)
    # Loop through all unique product/location pairs
    for product in df['Product'].unique():
        for location in df['Location'].unique():
            try:
                print(suggest_reorder(df, clf, product, location))
            except Exception as e:
                print(f"Skipping {product} at {location}: {e}")

