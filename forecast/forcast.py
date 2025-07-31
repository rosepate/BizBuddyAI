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



def create_sequences(data, sequence_length=10):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])
    return np.array(X), np.array(y)


product_sequences = {}

for product in products:
    product_df = df[df['Product'] == product]
    daily_sales = product_df.groupby('Date')['Units_Sold'].sum().reset_index().sort_values('Date')
    if len(daily_sales) <= 10:
        continue
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_units = scaler.fit_transform(daily_sales[['Units_Sold']])
    X, y = create_sequences(scaled_units.flatten(), sequence_length=10)
    X = X.reshape((X.shape[0], X.shape[1], 1))  # Reshape for LSTM input
    # Store the sequences and scaler for each product

    product_sequences[product] = {
        'X': X,
        'y': y,
        'scaler': scaler,
        'dates': daily_sales['Date'].values
    }
    plt.figure(figsize=(10, 4))
    plt.plot(daily_sales['Date'], daily_sales['Units_Sold'])
    plt.title(f'Daily Sales of {product}')
    plt.xlabel('Date')
    plt.ylabel('Units Sold')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

for product, data in product_sequences.items():
    print(f"{product}: X shape {data['X'].shape}, y shape {data['y'].shape}")

# Before creating the sequences, we need to scale the data to a range between 0 and 1. 
# This is a common practice for LSTM models as it helps with model convergence. We'll use MinMaxScaler from sklearn.preprocessing to achieve this.

from sklearn.preprocessing import MinMaxScaler

# Initialize the scaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Create a copy to avoid SettingWithCopyWarning
scaled_data = daily_sales.copy()

# Scale the 'Units_Sold' column
scaled_data['Units_Sold'] = scaler.fit_transform(daily_sales[['Units_Sold']])

# 3. Build and Train the LSTM Model
# Now that the data is prepared, we can build and train the LSTM model. 
# We'll split the data into training and testing sets, define the model architecture, compile it, and then train it.

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, batch_size=1, epochs=25, validation_data=(X_test, y_test))

# 4. Evaluate Model Performance
# After training, we'll visualize the model's performance by plotting the training and validation loss.

plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Increase model complexity
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(LSTM(64))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

history = model.fit(X, y, epochs=50, batch_size=16, validation_split=0.1, verbose=1)

# Add new column
daily_sales['DayOfWeek'] = pd.to_datetime(daily_sales['Date']).dt.dayofweek

# Normalize and add to input features
scaler_units = MinMaxScaler()
scaler_dow = MinMaxScaler()
scaled_units = scaler_units.fit_transform(daily_sales[['Units_Sold']])
scaled_dow = scaler_dow.fit_transform(daily_sales[['DayOfWeek']])

# Stack both features for multivariate input
multivariate_data = np.hstack((scaled_units, scaled_dow))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Build a deeper LSTM model
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(1))

# Compile
model.compile(optimizer='adam', loss='mean_squared_error')

# Optional: early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train
history = model.fit(
    X, y,
    epochs=50,
    batch_size=16,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

# Plot training history
plt.figure(figsize=(10, 4))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Enhanced LSTM Model Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.legend()
plt.grid(True)
plt.show()

# Forecast 7 days ahead
last_sequence = X[-1]
forecast = []
sequence_length=10

for _ in range(7):
    pred = model.predict(last_sequence.reshape(1, sequence_length, 1), verbose=0)
    forecast.append(pred[0][0])
    last_sequence = np.append(last_sequence[1:], pred).reshape(sequence_length, 1)

# Inverse scale
forecast_units = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))

import pandas as pd

# Dates for forecast
last_date = pd.to_datetime(scaled_data['Date'].iloc[-1])
forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=7)

# Plot
plt.figure(figsize=(12, 5))
plt.plot(daily_sales['Date'][-30:], daily_sales['Units_Sold'][-30:], label='Actual Sales', marker='o')
plt.plot(forecast_dates, forecast_units, label='Enhanced Forecast (7 Days)', marker='x', linestyle='--', color='orange')
plt.title("Enhanced Pharmacy Sales Forecast - LSTM")
plt.xlabel("Date")
plt.ylabel("Units Sold")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()