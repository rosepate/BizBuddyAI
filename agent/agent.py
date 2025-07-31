import streamlit as st
import sys
import os
import pandas as pd
from dotenv import load_dotenv

sys.path.append(r'c:\Users\rozyp\OneDrive\Desktop\Bizbuddy\BizBuddyAI')

from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.memory import ConversationBufferMemory

from forecast.forecasting import get_sales_forecast, product_location_sequences
from forecast.anomaly import load_data as load_anomaly_data, detect_z_score_anomalies
from forecast.auto_reorder_ml import load_data as load_reorder_data, create_features, train_reorder_model, suggest_reorder

def load_agent():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = api_key

    sheet_url = "https://docs.google.com/spreadsheets/d/1ISS7IQOMPrAEqU7lnpJYM5W2zd4oynntnmMTiokiVNU/export?format=csv"
    df = pd.read_csv(sheet_url)
    if "Order Date" in df.columns:
        df.rename(columns={"Order Date": "Date"}, inplace=True)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
    required_cols = ['Date', 'Product', 'Category', 'Units_Sold', 'Inventory_After', 'Location', 'Platform', 'Payment_Method', 'Product_Expiry_Date']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    agent = create_pandas_dataframe_agent(
        llm, df, verbose=True,
        memory=memory,
        handle_parsing_errors=True,
        agent_type="openai-tools",
        allow_dangerous_code=True,
    )
    return agent

agent = load_agent()

# Train reorder model ONCE
_reorder_df = create_features(load_reorder_data())
_reorder_clf = train_reorder_model(_reorder_df)

def agent_respond(user_query):
    # Forecast
    if "forecast" in user_query.lower():
        for (product, location) in product_location_sequences.keys():
            if product.lower() in user_query.lower() and location.lower() in user_query.lower():
                try:
                    dates, units = get_sales_forecast(product, location)
                    forecast_str = "\n".join([f"{d.date()}: {int(u[0]) if hasattr(u, '__iter__') else int(u)} units" for d, u in zip(dates, units)])
                    return f"📈 7-day sales forecast for {product} at {location}:\n{forecast_str}"
                except Exception as e:
                    return f"Sorry, could not generate forecast for {product} at {location}: {e}"
        return "Please specify both a valid product and location for forecasting."

    # Anomaly detection
    if "anomaly" in user_query.lower():
        df = load_anomaly_data()
        product_location_pairs = {(row['Product'], row['Location']) for _, row in df.iterrows()}
        for (product, location) in product_location_pairs:
            if product.lower() in user_query.lower() and location.lower() in user_query.lower():
                filtered = df[(df["Product"] == product) & (df["Location"] == location)]
                if filtered.empty:
                    return f"No data for {product} at {location}."
                result = []
                for col in ['Units_Sold', 'Inventory_After']:
                    if col in filtered.columns:
                        anomalies = detect_z_score_anomalies(filtered, column=col, threshold=3)
                        detected = anomalies[anomalies['Anomaly']]
                        if not detected.empty:
                            result.append(f"Anomalies in {col}:\n" + detected[['Date', col, 'z_score']].to_string(index=False))
                        else:
                            result.append(f"No anomalies detected in {col}.")
                return "\n\n".join(result)
        return "Please specify both a valid product and location for anomaly detection."

    # Reorder suggestion
    if "reorder" in user_query.lower():
        for product in _reorder_df['Product'].unique():
            for location in _reorder_df['Location'].unique():
                if product.lower() in user_query.lower() and location.lower() in user_query.lower():
                    return suggest_reorder(_reorder_df, _reorder_clf, product, location)
        return "Please specify both a valid product and location for reorder suggestion."

    # Fallback to agent
    try:
        response = agent.invoke(user_query)
        return response
    except Exception as e:
        return f"Agent error: {e}"

def chatbot_view(agent):
    st.title("💬 BizBuddy AI Chatbot")
    st.markdown("Chat naturally with your business data.")

    if "agent" not in st.session_state:
        st.session_state.agent = agent

    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.agent = agent
        st.rerun()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(message)

    user_input = st.chat_input("Ask your question...")

    if user_input:
        st.session_state.chat_history.append(("user", user_input))
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = agent_respond(user_input)
                    st.markdown(response)
                    st.session_state.chat_history.append(("assistant", response))
                except Exception as e:
                    st.error(f"⚠️ Error: {str(e)}")
