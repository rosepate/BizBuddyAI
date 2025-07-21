import streamlit as st
import plotly.express as px
import pandas as pd
import pdfkit
import tempfile
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def forecast_sales(df):
           st.subheader("📈 Sales Forecast")
           try:
               sales = df.groupby(pd.Grouper(key="Date", freq="M"))["Revenue"].sum().values
               scaler = MinMaxScaler()
               sales_scaled = scaler.fit_transform(sales.reshape(-1, 1))
               X, y = [], []
               for i in range(3, len(sales_scaled)):
                   X.append(sales_scaled[i-3:i])
                   y.append(sales_scaled[i])
               X, y = np.array(X), np.array(y)

               model = Sequential([
                   LSTM(50, activation="relu", input_shape=(3, 1)),
                   Dense(1)
               ])
               model.compile(optimizer="adam", loss="mse")
               model.fit(X, y, epochs=50, verbose=0)

               last_3 = sales_scaled[-3:]
               pred = model.predict(np.array([last_3]), verbose=0)
               pred_unscaled = scaler.inverse_transform(pred)[0][0]
               st.write(f"Predicted Revenue for Next Month: ${pred_unscaled:,.2f}")
           except Exception as e:
               st.warning(f"Error in sales forecasting: {e}")

def detect_anomalies(df):
           st.subheader("🚨 Anomaly Detection")
           try:
               features = df[["Revenue", "Units_Sold", "Inventory_After"]].dropna()
               model = IsolationForest(contamination=0.1, random_state=42)
               model.fit(features)
               df["Anomaly"] = model.predict(features)
               anomalies = df[df["Anomaly"] == -1][["Date", "Product", "Revenue", "Units_Sold"]]
               if not anomalies.empty:
                   st.dataframe(anomalies)
               else:
                   st.write("No anomalies detected.")
           except Exception as e:
               st.warning(f"Error in anomaly detection: {e}")

def export_to_pdf(df):
           st.subheader("📄 Export Dashboard as PDF")
           try:
               html = f"<h1>BizBuddy AI Dashboard</h1><p>Generated on {pd.Timestamp.today()}</p>"
               html += df.to_html()
               with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
                   tmp.write(html.encode())
                   tmp_path = tmp.name
               pdf_path = "dashboard.pdf"
               pdfkit.from_file(tmp_path, pdf_path)
               with open(pdf_path, "rb") as f:
                   st.download_button("Download Dashboard as PDF", f, "dashboard.pdf", "application/pdf")
               os.remove(tmp_path)
               os.remove(pdf_path)
           except Exception as e:
               st.error(f"Error generating PDF: {e}")

def dashboard_view(df):
           st.title("BizBuddy AI Dashboard")
           st.markdown("Key metrics and trends for your pharmacy business.")

           required_cols = ["Date", "Product", "Revenue", "Units_Sold", "Inventory_After", "Location"]
           optional_cols = ["Expiry Date", "Platform"]
           missing_cols = [col for col in required_cols if col not in df.columns]
           if missing_cols:
               st.error(f"Missing required columns: {', '.join(missing_cols)}")
               return

           try:
               df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
               if "Expiry Date" in df.columns:
                   df["Expiry Date"] = pd.to_datetime(df["Expiry Date"], errors="coerce")
           except Exception as e:
               st.error(f"Date conversion error: {e}")
               return

           numeric_cols = ["Revenue", "Units_Sold", "Inventory_After", "Unit_Price", "Cost_Price", "Profit"]
           for col in numeric_cols:
               if col in df.columns:
                   try:
                       df[col] = pd.to_numeric(df[col], errors="coerce")
                   except Exception as e:
                       st.error(f"Error converting {col} to numeric: {e}")
                       return

           st.subheader("Key Performance Indicators")
           col1, col2, col3, col4 = st.columns(4)
           total_revenue = df["Revenue"].sum()
           total_units = df["Units_Sold"].sum()
           top_product = df.groupby("Product")["Revenue"].sum().idxmax() if not df["Product"].empty else "N/A"
           top_location = df.groupby("Location")["Revenue"].sum().idxmax() if not df["Location"].empty else "N/A"

           with col1:
               st.metric("💵 Total Revenue", f"${total_revenue:,.2f}" if pd.notna(total_revenue) else "N/A")
           with col2:
               st.metric("📦 Total Units Sold", f"{total_units:,.0f}" if pd.notna(total_units) else "N/A")
           with col3:
               st.metric("🏆 Top Product", top_product)
           with col4:
               st.metric("📍 Top Location", top_location)

           st.subheader("📈 Monthly Revenue Trend")
           try:
               monthly = df.groupby(pd.Grouper(key="Date", freq="M"))["Revenue"].sum().reset_index()
               if not monthly.empty:
                   chart_data = monthly.rename(columns={"Date": "Month"}).set_index("Month")
                   st.line_chart(chart_data)
               else:
                   st.warning("No data available for monthly revenue trend.")
           except Exception as e:
               st.warning(f"Error rendering monthly revenue trend: {e}")

           st.subheader("📊 Revenue by Product")
           try:
               product_revenue = df.groupby("Product")["Revenue"].sum().sort_values(ascending=False)
               if not product_revenue.empty:
                   st.bar_chart(product_revenue)
               else:
                   st.warning("No data available for revenue by product.")
           except Exception as e:
               st.warning(f"Error rendering revenue by product: {e}")

           if "Platform" in df.columns:
               st.subheader("🛍️ Revenue by Platform")
               try:
                   platform_revenue = df.groupby("Platform")["Revenue"].sum().sort_values(ascending=False)
                   if not platform_revenue.empty:
                       st.bar_chart(platform_revenue)
                   else:
                       st.warning("No data available for revenue by platform.")
               except Exception as e:
                   st.warning(f"Error rendering revenue by platform: {e}")
           else:
               st.info("Platform data not available.")

           st.subheader("🗺️ Revenue by Location")
           try:
               location_revenue = df.groupby("Location")["Revenue"].sum().reset_index()
               if not location_revenue.empty:
                   fig = px.bar(location_revenue, x="Location", y="Revenue", color="Location", title="Revenue by Location")
                   st.plotly_chart(fig)
               else:
                   st.warning("No data available for revenue by location.")
           except Exception as e:
               st.warning(f"Error rendering revenue by location: {e}")

           st.subheader("📦 Products with Low Inventory")
           try:
               low_stock = df[df["Inventory_After"] < 20].groupby("Product")["Inventory_After"].min().sort_values().reset_index().head(10)
               if not low_stock.empty:
                   st.dataframe(low_stock)
                   st.warning("Alert: Low inventory detected!")
               else:
                   st.write("No products with low inventory.")
           except Exception as e:
               st.warning(f"Error rendering low inventory alerts: {e}")

           if "Expiry Date" in df.columns:
               st.subheader("⏰ Upcoming Expiry Medicines (Next 60 Days)")
               try:
                   exp_soon = df[df["Expiry Date"] <= pd.Timestamp("2025-07-21") + pd.Timedelta(days=60)]
                   exp_soon_display = exp_soon[["Product", "Expiry Date", "Inventory_After"]].drop_duplicates()
                   if not exp_soon_display.empty:
                       st.dataframe(exp_soon_display)
                       st.warning("Alert: Expiry risk detected!")
                   else:
                       st.write("No products expiring within the next 60 days.")
               except Exception as e:
                   st.warning(f"Error rendering expiry alerts: {e}")
           else:
               st.info("Expiry date data not available.")

           if st.button("Run Sales Forecast"):
               forecast_sales(df)

           if st.button("Detect Anomalies"):
               detect_anomalies(df)

           export_to_pdf(df)

           st.subheader("📊 Power BI Integration")
           st.write("Download the dataset and import it into Power BI for advanced visualizations.")
           try:
               csv = df.to_csv(index=False)
               st.download_button(
                   label="Download Data for Power BI",
                   data=csv,
                   file_name="powerbi_data.csv",
                   mime="text/csv",
               )
           except Exception as e:
               st.error(f"Error generating Power BI CSV: {e}")

           st.subheader("🎯 Top Profit Contributors")
           try:
               top_profit = df.groupby("Product")["Profit"].sum().sort_values(ascending=False).head(3)
               if not top_profit.empty:
                   st.bar_chart(top_profit)
                   st.write("Top 3 Profit Contributors:", top_profit.index.tolist())
               else:
                   st.warning("No data available for profit contributors.")
           except Exception as e:
               st.warning(f"Error rendering top profit contributors: {e}")

           st.subheader("🤖 Product Recommendations")
           if st.button("Generate Recommendations"):
               try:
                   top_products = df.groupby("Product")["Revenue"].sum().sort_values(ascending=False).head(3).index
                   st.write(f"Recommended Products: {', '.join(top_products)} based on revenue.")
               except Exception as e:
                   st.warning(f"Error generating recommendations: {e}")

           st.subheader("🏅 High/Low Performers")
           try:
               perf = df.groupby("Product")["Profit"].mean().sort_values()
               low_performers = perf.head(3).index.tolist()
               high_performers = perf.tail(3).index.tolist()
               st.write("Low Performers:", low_performers)
               st.write("High Performers:", high_performers)
           except Exception as e:
               st.warning(f"Error rendering performers: {e}")

           st.subheader("🛒 Auto Reorder Suggestions")
           try:
               reorder = df[df["Inventory_After"] < 30].groupby("Product")["Inventory_After"].min().sort_values()
               if not reorder.empty:
                   st.dataframe(reorder.reset_index())
                   st.write("Suggestion: Reorder these products.")
               else:
                   st.write("No reorder suggestions needed.")
           except Exception as e:
               st.warning(f"Error rendering reorder suggestions: {e}")

           st.subheader("⏳ Stock Expiry Risk Model")
           if "Expiry Date" in df.columns:
               try:
                   risk = df[df["Expiry Date"] <= pd.Timestamp("2025-07-21") + pd.Timedelta(days=90)]
                   if not risk.empty:
                       st.dataframe(risk[["Product", "Expiry Date", "Inventory_After"]])
                       st.warning("High expiry risk detected!")
                   else:
                       st.write("No high expiry risk.")
               except Exception as e:
                   st.warning(f"Error rendering expiry risk: {e}")