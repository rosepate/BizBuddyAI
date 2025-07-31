import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def load_data():
    sheet_url = "https://docs.google.com/spreadsheets/d/1N0cGSLkm9k-p5aIDsPbVp-fOl5SZVitJfQLs9LThANk/export?format=csv"
    df = pd.read_csv(sheet_url)
    if "Sale Date" in df.columns:
        df.rename(columns={"Sale Date": "Date"}, inplace=True)
    return df

df = load_data()
products = df['Product'].unique()
locations = df['Location'].unique()

def create_sequences(data, sequence_length=10):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])
    return np.array(X), np.array(y)

# Build product_location_sequences
product_location_sequences = {}

for product in products:
    for location in locations:
        product_df = df[(df['Product'] == product) & (df['Location'] == location)]
        if product_df.empty:
            continue
        product_df = product_df.copy()
        product_df['Date'] = pd.to_datetime(product_df['Date'])
        daily_sales = product_df.groupby('Date')['Units_Sold'].sum().reset_index().sort_values('Date')
        if len(daily_sales) <= 10:
            continue
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_units = scaler.fit_transform(daily_sales[['Units_Sold']])
        X, y = create_sequences(scaled_units.flatten(), sequence_length=10)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        product_location_sequences[(product, location)] = {
            'X': X,
            'y': y,
            'scaler': scaler,
            'dates': daily_sales['Date'].values,
            'daily_sales': daily_sales
        }

for (product, location), data in product_location_sequences.items():
    print(f"{product} @ {location}: X shape {data['X'].shape}, y shape {data['y'].shape}")

# --- FORECAST FUNCTION FOR DASHBOARD/CHATBOT USE ---
def get_sales_forecast(product_name, location, plot=False):
    """
    Returns forecast_dates, forecast_units for the given product_name and location.
    If plot=True, also displays the forecast plot.
    """
    key = (product_name, location)
    if key not in product_location_sequences:
        raise ValueError(f"No data for product '{product_name}' at location '{location}'.")

    X = product_location_sequences[key]['X']
    y = product_location_sequences[key]['y']
    scaler = product_location_sequences[key]['scaler']
    daily_sales = product_location_sequences[key]['daily_sales'].copy()
    daily_sales['Date'] = pd.to_datetime(daily_sales['Date'])

    from sklearn.model_selection import train_test_split
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=1, epochs=5, validation_data=(X_test, y_test), verbose=0)

    # Enhanced model
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(LSTM(64))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model.fit(X, y, epochs=10, batch_size=16, validation_split=0.1, callbacks=[early_stop], verbose=0)

    # Forecast
    last_sequence = X[-1]
    forecast = []
    sequence_length = X.shape[1]
    for _ in range(7):
        pred = model.predict(last_sequence.reshape(1, sequence_length, 1), verbose=0)
        forecast.append(pred[0][0])
        last_sequence = np.append(last_sequence[1:], pred).reshape(sequence_length, 1)
    forecast_units = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
    last_date = daily_sales['Date'].iloc[-1]
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=7)

    if plot:
        plt.figure(figsize=(12, 5))
        plt.plot(daily_sales['Date'][-30:], daily_sales['Units_Sold'][-30:], label='Actual Sales', marker='o')
        plt.plot(forecast_dates, forecast_units, label='Enhanced Forecast (7 Days)', marker='x', linestyle='--', color='orange')
        plt.title(f"Enhanced Sales Forecast - LSTM ({product_name} @ {location})")
        plt.xlabel("Date")
        plt.ylabel("Units Sold")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return forecast_dates, forecast_units

# --- OPTIONAL: Interactive Widget for Notebooks ---
if __name__ == "__main__":
    try:
        import ipywidgets as widgets
        from IPython.display import display, clear_output

        product_list = sorted(set([k[0] for k in product_location_sequences.keys()]))
        location_dict = {}
        for prod in product_list:
            location_dict[prod] = sorted([loc for (p, loc) in product_location_sequences.keys() if p == prod])

        product_dropdown = widgets.Dropdown(
            options=product_list,
            description='Product:',
            value=product_list[0]
        )

        location_dropdown = widgets.Dropdown(
            options=location_dict[product_list[0]],
            description='Location:',
            value=location_dict[product_list[0]][0]
        )

        def on_product_change(change):
            location_dropdown.options = location_dict[change['new']]
            location_dropdown.value = location_dict[change['new']][0]
            clear_output(wait=True)
            display(product_dropdown, location_dropdown)
            get_sales_forecast(product_dropdown.value, location_dropdown.value, plot=True)

        def on_location_change(change):
            clear_output(wait=True)
            display(product_dropdown, location_dropdown)
            get_sales_forecast(product_dropdown.value, location_dropdown.value, plot=True)

        product_dropdown.observe(on_product_change, names='value')
        location_dropdown.observe(on_location_change, names='value')

        display(product_dropdown, location_dropdown)
        get_sales_forecast(product_dropdown.value, location_dropdown.value, plot=True)
    except ImportError:
        print("ipywidgets not installed. Run this script in a notebook for interactive selection.")