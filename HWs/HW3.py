from openai import OpenAI
import streamlit as st
import tiktoken
import requests
from bs4 import BeautifulSoup



st.title ("Chatty G - Lab 3: Streamlit Chat Interface")

# Below is the code to set up OpenAI client and default model - pull responses from secrets

# Set OpenAI API key from Streamlit secrets
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Set a default model and max tokens for the chat completions
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4-turbo"


# URL Reader (provided by Prof.)
def read_url_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status() # raise an eception for HTTP errors
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup.get_text(separator='\n')
    except requests.RequestException as e:
        st.error(f"Error reading {url}: {e}")
        return None

# SIDEBAR URL Input
with st.sidebar:
    st.header("Sources")

    url1 = st.text_input("URL 1", placeholder="https://example.com/article")
    url2 = st.text_input("URL 2 (optional)", placeholder="https://example.com/another")

    load_urls = st.button("Load URL(s)")

# INITIALIZE storage for URL text
if "url_context" not in st.session_state:
    st.session_state.url_context = ""

if load_urls:
    texts = []

    # Basic VALIDATION (avoid fetching empty strings)
    if url1.strip():
        t1 = read_url_content(url1.strip())
        if t1:
            texts.append(f"SOURCE 1 ({url1}):\N{t1}")

    if url2.strip():
        t2 = read_url_content(url2.strip())
        if t2:
            texts.append(f"SOURCE 2 ({url2}):\n{t2}")

        if texts:
            # COMBINE and STORE for later use in the chatbot prompt
            st.session_state.url_context = "\n\n".join(texts)
            st.sidebar.success("Loaded URL content!")
        else:
            st.session_state.url_context = ""
            st.sidebar.warning("No valid URL contnt loaded.")

# Below is the code for a simple chat interface using Streamlit's chat components
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system", 
            "content": (
                "You are Chatty G, a helpful and friendly assistant."
                
            )
        }   
    ]
enc = tiktoken.encoding_for_model(st.session_state["openai_model"])

# Token limit for input messages shown in progress bar
MAX_TOKENS_IN = 900

# Token counter without tiktoken
def estimate_tokens(messages) -> int:
    words = sum(len((m.get("content") or "").split()) for m in messages)
    return int(words * 1.3)  # Estimate tokens as 1.3x the number of words

# A function to count tokens in messages
def tok(messages):
    return sum(len(enc.encode(m.get("role","") + (m.get("content","") or ""))) for m in messages)

# Ensure system message is kept
def build_context():
    sys_msg = [m for m in st.session_state.messages if m["role"] == "system"]
    # keep last 4 messages from chat history.
    chat = [m for m in st.session_state.messages if m["role"] != "system"][-4:]
    context = sys_msg + chat

    # Remove oldest messages until within token limit
    while len(chat) > 0 and tok(context) > MAX_TOKENS_IN:
        chat.pop(0)
        context = sys_msg + chat
    return context


# Display chat messages from history on app rerun 
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What would you like to ask Chatty G?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    context = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]

    tokens_used = estimate_tokens(context)
    pct = int((tokens_used / MAX_TOKENS_IN) * 100) if MAX_TOKENS_IN else 0
    pct = max(0, min(pct, 100))  # Ensure percentage is between 0 and 100

    with st.sidebar:
        st.subheader("Token Usage")
        st.progress(pct, text=f"{tokens_used} / {MAX_TOKENS_IN} tokens (estimate)")
        

# Display assistant response in chat message container (streaming)
with st.chat_message("assistant"):
     stream = client.chat.completions.create(
         model=st.session_state["openai_model"],
         messages=[
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
         ],
         stream=True,
     )
     response = st.write_stream(stream)
st.session_state.messages.append({"role": "assistant", "content": response})        
   



    





