import streamlit as st
import pandas as pd
import os
import sys
from dotenv import load_dotenv
from models import SalesRecord
from pydantic import ValidationError

st.set_page_config(page_title="BizBuddy AI", page_icon="🧠", layout="wide")
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

@st.cache_data(ttl=60)
def load_data():
           sheet_url = "https://docs.google.com/spreadsheets/d/1ISS7IQOMPrAEqU7lnpJYM5W2zd4oynntnmMTiokiVNU/export?format=csv"
           try:
               df = pd.read_csv(sheet_url)
               if "Sale Date" in df.columns:
                   df.rename(columns={"Sale Date": "Date"}, inplace=True)
               if "Product_Expiry_Date" in df.columns:
                   df.rename(columns={"Product_Expiry_Date": "Expiry Date"}, inplace=True)
               if "Units_Sold" in df.columns:
                   df.rename(columns={"Units_Sold": "Units Sold"}, inplace=True)
               if "Inventory_After" in df.columns:
                   df.rename(columns={"Inventory_After": "Inventory After"}, inplace=True)
               if "Unit_Price" in df.columns:
                   df.rename(columns={"Unit_Price": "Unit Price"}, inplace=True)
               if "Cost_Price" in df.columns:
                   df.rename(columns={"Cost_Price": "Cost Price"}, inplace=True)
               validated_data = []
               for _, row in df.iterrows():
                   try:
                       record = SalesRecord(**row.to_dict())
                       validated_data.append(record.dict())
                   except ValidationError as e:
                       st.error(f"Validation error for row: {e}")
                       return pd.DataFrame()
               return pd.DataFrame(validated_data)
           except Exception as e:
               st.error(f"Error loading data: {e}")
               return pd.DataFrame()

df = load_data()
from agent.agent import load_agent
agent = load_agent()

from chat.streamlit_chats import chatbot_view
from dashboard.streamlit_dashboards import dashboard_view

st.title("BizBuddy AI")
st.markdown("Welcome to BizBuddy AI! Use the navigation menu to explore the chatbot or dashboard.")

st.sidebar.title("🧽 Navigation")
page = st.sidebar.radio("Go to:", ["💬 Chatbot", "📊 Dashboard"])

if page == "💬 Chatbot":
   chatbot_view(agent)
elif page == "📊 Dashboard":
   dashboard_view(df)