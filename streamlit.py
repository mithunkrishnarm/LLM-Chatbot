import streamlit as st
import requests
import json

st.set_page_config(page_title="GeminiSpeak", page_icon="💬", layout="centered")  # Only once, at the top

# --- Custom CSS for Stylish Design ---
st.markdown("""
    <style>
    .stChatMessage {
        border-radius: 16px;
        padding: 12px 18px;
        margin-bottom: 10px;
        font-size: 1.1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .stChatMessage.user {
        background: linear-gradient(90deg, #e0e7ff 0%, #f3f4f6 100%);
        color: #22223b;
        align-self: flex-end;
    }
    .stChatMessage.assistant {
        background: linear-gradient(90deg, #f9fafb 0%, #e0f2fe 100%);
        color: #1e293b;
        align-self: flex-start;
    }
    .stButton>button {
        background: #6366f1;
        color: white;
        border-radius: 8px;
        font-weight: 600;
        border: none;
        padding: 0.5em 1.2em;
        margin-top: 10px;
    }
    .stButton>button:hover {
        background: #4338ca;
        color: #fff;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- Configuration ---
FASTAPI_URL = "http://localhost:8001"
CHAT_ENDPOINT = f"{FASTAPI_URL}/chat"

# --- Streamlit UI Setup ---
st.markdown("<h1 style='text-align:center; color:#6366f1;'>💬 Mithun Kichu's AI Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#64748b;'>Ask anything! I am here to help you.</p>", unsafe_allow_html=True)

# Sidebar styling and clear button
with st.sidebar:
    st.markdown("<h2 style='color:#6366f1;'>Options</h2>", unsafe_allow_html=True)
    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Initialize chat history in Streamlit's session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Display Chat Messages with Avatars and Styling ---
# Iterate through messages and display them with appropriate avatars
for message in st.session_state.messages:
    avatar = "🧑‍💻" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"]):
        st.markdown(
            f"<div class='stChatMessage {message['role']}'>"
            f"<span style='font-size:1.3em;'>{avatar}</span> &nbsp; {message['content']}"
            f"</div>",
            unsafe_allow_html=True
        )


# --- Chat Input with UI Feedback ---
# We'll use a placeholder for the chat input text
prompt = st.chat_input("Say something...", disabled=st.session_state.get("thinking", False))
# `disabled` state will be managed when AI is responding.

if prompt:
    # 1. Add user message to chat history and display immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(
            f"<div class='stChatMessage user'>🧑‍💻 &nbsp; {prompt}</div>",
            unsafe_allow_html=True
        )

    # 2. Prepare chat history for FastAPI
    chat_history_for_api = []
    for msg in st.session_state.messages:
        if msg["role"] in ("user", "assistant"):
            chat_history_for_api.append({
                "role": msg["role"],
                "parts": [{"text": msg["content"]}]
            })

    # --- Communicate with FastAPI Backend with Enhanced Error Handling & UI Feedback ---
    try:
        # Set a flag to disable input and show spinner
        st.session_state.thinking = True
        with st.spinner("🤖 Chatbot is thinking..."):
            response = requests.post(
                CHAT_ENDPOINT,
                json={"message": prompt, "chat_history": chat_history_for_api},
                timeout=60 # Add a timeout to prevent indefinite waiting
            )
            response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)

            # Parse the JSON response
            ai_response_data = response.json()
            ai_message = ai_response_data.get("ai_message", "Sorry, I couldn't get a clear response from Gemini.")

        # 3. Add AI message to chat history and display
        st.session_state.messages.append({"role": "assistant", "content": ai_message})
        with st.chat_message("assistant"):
            st.markdown(
                f"<div class='stChatMessage assistant'>🤖 &nbsp; {ai_message}</div>",
                unsafe_allow_html=True
            )

    except requests.exceptions.ConnectionError:
        st.error("Connection Error: Could not connect to the backend. Please ensure the FastAPI server is running and accessible.")
    except requests.exceptions.Timeout:
        st.error("Request Timeout: The backend took too long to respond. Please try again.")
    except requests.exceptions.HTTPError as e:
        # More detailed error handling for HTTP responses
        status_code = e.response.status_code
        error_detail = "An unknown error occurred."
        try:
            error_json = e.response.json()
            error_detail = error_json.get("detail", error_detail)
        except json.JSONDecodeError:
            error_detail = e.response.text # Fallback to raw text if not JSON

        if status_code == 400:
            st.warning(f"Bad Request (Code 400): {error_detail}. Your prompt might have been too long or violated safety policies.")
        elif status_code == 404:
            st.error(f"Not Found (Code 404): {error_detail}. The backend endpoint might be incorrect or missing.")
        elif status_code == 500:
            st.error(f"Internal Server Error (Code 500): {error_detail}. The backend encountered a problem.")
        else:
            st.error(f"HTTP Error {status_code}: {error_detail}")
    except json.JSONDecodeError:
        st.error("Invalid Response: Received an unreadable response from the backend. The server might be misconfigured.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
    finally:
        # Always reset the thinking flag
        st.session_state.thinking = False
        # Re-run Streamlit to update the UI (e.g., re-enable input)
        st.rerun() # Ensure input is re-enabled if an error occurs