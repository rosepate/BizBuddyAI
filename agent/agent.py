from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent  # <-- updated import
from langchain.memory import ConversationBufferMemory
import pandas as pd
import os
from dotenv import load_dotenv

import sys
sys.path.append(r'c:\Users\rozyp\OneDrive\Desktop\Bizbuddy\BizBuddyAI')

# 1. Import your forecasting function
from forecast.forecasting import get_sales_forecast, product_location_sequences

def load_agent():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = api_key

    # Load from Google Sheet CSV

    sheet_url = "https://docs.google.com/spreadsheets/d/1ISS7IQOMPrAEqU7lnpJYM5W2zd4oynntnmMTiokiVNU/export?format=csv"
    df = pd.read_csv(sheet_url)
    #print column namesuv pip install -r requirements.txt

    print("📊 DataFrame loaded with columns:", df.columns.tolist()) 

# 🛡️ Handle date column gracefully
    if "Order Date" in df.columns:
        df.rename(columns={"Order Date": "Date"}, inplace=True)

    if "Date" in df.columns:
        try:
            df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
        except Exception as e:
            print("⚠️ Date conversion error:", e)

    # ✅ Check for important columns needed for analytics
    required_cols = ['Date', 'Product', 'Category', 'Units_Sold', 'Inventory_After', 'Location', 'Platform', 'Payment_Method', 'Product_Expiry_Date', ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"⚠️ Missing key columns: {missing_cols}")

    # 🧾 Show columns and preview data
    print("🧾 Columns in dataset:", df.columns.tolist())
    print("🔍 Sample rows:\n", df.head())
    # ✅ Ensure all required columns are present
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    # ✅ Print first few rows for debugging
    print("🔍 Sample data:\n", df.head()
        )
    
    # ✅ Print column list for debugging
    print("🧾 Columns in dataset:", df.columns.tolist())

    # Set up the agent
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        memory=memory,
        handle_parsing_errors=True,
        agent_type="openai-tools",
        allow_dangerous_code=True,
    )

    return agent

agent = load_agent()

# 2. Add a function to handle user queries, including forecasting
def agent_respond(user_query):
    if "forecast" in user_query.lower():
        for (product, location) in product_location_sequences.keys():
            if product.lower() in user_query.lower() and location.lower() in user_query.lower():
                try:
                    dates, units = get_sales_forecast(product, location)
                    # Handle both 1D and 2D units
                    try:
                        forecast_str = "\n".join([f"{d.date()}: {int(u[0])} units" for d, u in zip(dates, units)])
                    except Exception:
                        forecast_str = "\n".join([f"{d.date()}: {int(u)} units" for d, u in zip(dates, units)])
                    return f"📈 7-day sales forecast for {product} at {location}:\n{forecast_str}"
                except Exception as e:
                    return f"Sorry, could not generate forecast for {product} at {location}: {e}"
        return "Please specify both a valid product and location for forecasting."
    else:
        try:
            response = agent.invoke(user_query)
            return response
        except Exception as e:
            return f"Agent error: {e}"

if __name__ == "__main__":
    while True:
        user_query = input("Ask your question (type 'exit' to quit): ")
        if user_query.lower() == "exit":
            break
        answer = agent_respond(user_query)
        print(answer)
