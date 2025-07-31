import streamlit as st
import sys
sys.path.append(r'c:\Users\rozyp\OneDrive\Desktop\Bizbuddy\BizBuddyAI')

from agent.agent import agent_respond, load_agent


def chatbot_view(agent):
    st.title("💬 BizBuddy AI Chatbot")
    st.markdown("Chat naturally with your business data.")

    # Initialize agent in session state only once
    if "agent" not in st.session_state:
        st.session_state.agent = load_agent()

    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        # Optionally reload agent if you want a fresh memory
        st.session_state.agent = load_agent()
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
                    # Pass the agent instance if needed
                    response = agent_respond(user_input)
                    st.markdown(response)
                    st.session_state.chat_history.append(("assistant", response))
                except Exception as e:
                    st.error(f"⚠️ Error: {str(e)}")
    
from forecast.anomaly import detect_z_score_anomalies
def agent_respond(user_query):
    # ...existing forecast logic...
    if "anomaly" in user_query.lower():
        # Try to extract product and location from the query
        from forecast.anomaly import load_data
        df = load_data()
        # Build product_location_sequences from the DataFrame
        product_location_sequences = {(row['Product'], row['Location']) for _, row in df.iterrows()}
        for (product, location) in product_location_sequences:
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
    # ...existing fallback logic...
    
