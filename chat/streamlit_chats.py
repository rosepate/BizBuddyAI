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
