import streamlit as st
import requests
import time

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Oncology AI Assistant",
    page_icon="⚕️",
    layout="centered"
)

st.title("⚕️ Oncology AI Assistant")
st.caption("AI-powered cancer knowledge assistant (context-based)")

API_URL = "http://192.168.56.1:8000/ask"

# =========================
# Session State
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "loading" not in st.session_state:
    st.session_state.loading = False

# =========================
# Chat History
# =========================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# =========================
# Chat Input
# =========================
prompt = st.chat_input(
    "Ask about cancer...",
    disabled=st.session_state.loading
)

if prompt:
    # User Message
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("🧠 Analyzing medical context..."):
            st.session_state.loading = True
            try:
                res = requests.post(
                    API_URL,
                    json={"question": prompt},
                    timeout=60
                )

                if res.status_code == 200:
                    data = res.json()
                    answer = data.get("answer", "No response received.")
                    st.markdown(answer)

                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )
                else:
                    st.error("⚠️ Backend returned an error.")

            except requests.exceptions.ConnectionError:
                st.error("🚫 Backend is not running. Start FastAPI server.")
            except requests.exceptions.Timeout:
                st.error("⏳ Backend took too long to respond.")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

            st.session_state.loading = False
