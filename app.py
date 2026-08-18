import streamlit as st
from Chatbot_M001WD import get_response
from google import genai
api_key = st.secrets["GEMINI_API_KEY"]
client = genai.Client(api_key=api_key)

st.title("WD World Chat-Buddy")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

suggestions = [
    "Hey...",
    "About You",
    "Contact Us",
    "What is AI?"
]

st.markdown("**Suggested Questions:**")
cols = st.columns(len(suggestions))
clicked_prompt = None

for idx, question in enumerate(suggestions):
    if cols[idx].button(question, key=f"btn_{idx}"):
        clicked_prompt = question

user_input = st.chat_input("Ask something...") or clicked_prompt

if user_input:
    st.session_state.messages.append({"role" : "user", "content" : user_input})

    reply = get_response(user_input)

    st.session_state.messages.append({"role" : "assistant", "content" : reply})
    st.rerun()
