import streamlit as st
from Components.home import Section as HomePage
from Components.SyntheticDatagenerator import Section as FirstSection
from Components.customModelTrainer import Section as SecondSection
from Components.metrics_final import Section as ThirdSection
import requests

if "page" not in st.session_state:
    st.session_state.page = 'Home'

# Sidebar options
with st.sidebar:
    st.header("Navigation")
    if st.button("Home"):
        st.session_state.page = "Home"
    if st.button("Generate synthetic Data"):
        st.session_state.page = "🧬 Synthetic Diabetes Data Generator"
    if st.button("Train custom model"):
        st.session_state.page = "🛠️ Train Custom Synthetic Data Model"
    if st.button("Evaluation & Metrics"):
        st.session_state.page = "📊 Evaluation & Metrics Center"
    st.markdown('---')
    # --- Assistant ---
    st.subheader("Tinyllama Assistant")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    user_input = st.text_input("Ask something:", key="chat_input")
    col1, col2 = st.columns([1, 1])

    with col2:
        if st.button("Clear", key="clr_btn"):
            st.session_state.chat_history = []

    with col1:
        if st.button("Send", key="send_btn"):
            if user_input:
                try:
                    response = requests.post(
                        "http://localhost:11434/api/generate",
                        json={"model": "tinyllama", "prompt": user_input, "stream": False}
                    )
                    reply = response.json()["response"]
                except Exception as e:
                    reply = f"⚠️ Error: {str(e)}"
                st.session_state.chat_history.append(("You", user_input))
                st.session_state.chat_history.append(("Assistant", reply))

    # Display last 5 messages
    for sender, msg in reversed(st.session_state.chat_history[-5:]):
        st.markdown(f"**{sender}:** {msg}")


st.title(st.session_state.page)
if st.session_state.page == "Home":
    HomePage()
if st.session_state.page == "🧬 Synthetic Diabetes Data Generator":
    FirstSection()
if st.session_state.page == "🛠️ Train Custom Synthetic Data Model":
    SecondSection()
if st.session_state.page == "📊 Evaluation & Metrics Center":
    ThirdSection()
    
    
#main page

