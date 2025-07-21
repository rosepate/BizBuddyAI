import streamlit as st

def chatbot_view(agent):
           st.title("💬 BizBuddy AI Chatbot")
           st.markdown("Chat naturally with your business data.")

           if st.button("🗑️ Clear Chat"):
               st.session_state.chat_history = []
               if "agent" in st.session_state:
                   del st.session_state.agent
               from agent.agent import load_agent
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
                           response = agent.run(user_input)
                           st.markdown(response)
                           st.session_state.chat_history.append(("assistant", response))
                       except Exception as e:
                           st.error(f"⚠️ Error: {str(e)}")