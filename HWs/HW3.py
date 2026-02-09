import streamlit as st
import requests
from bs4 import BeautifulSoup 
from openai import OpenAI
from anthropic import Anthropic 


# Name of APP and description

st.title("Chatty G")
st.write("An app that summarizes the contents of web pages.")

PERSONA = "You are Chatty G, a helpful and friendly assistant."

# SESSION STATE ------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []      # user/assistant only (display + buffer)

if "url_context" not in st.session_state: 
    st.session_state.url_context = ""  # combine text from URL(s)

if "summary" not in st.session_state:
    st.session_state.summary = "" # compress memory

if "provider" not in st.session_state:
    st.session_state.provider = "OpenAI"

if "openai_model" not in st.session_state:
    st.session_state.openai_model = "gpt-4.6"

if "claude_model" not in st.session_state:
    st.session_state.claude_model = "claude-opus-4-6"

# Client - secret keys in streamlit
openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
claude_client = Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])


# URL HELPERS -------------------------------
def read_url_content(url: str):
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        return soup.get_text(separator="\n")[:8000]  # cap size
    except requests.RequestException as e:
        st.error(f"Error reading {url}: {e}")
        return None

def load_urls(url1: str, url2: str):
    texts = []

    if url1.strip():
        t1 = read_url_content(url1.strip())
        if t1:
            texts.append(f"SOURCE 1 ({url1.strip()}):\n{t1}")

    if url2.strip():
        t2 = read_url_content(url2.strip())
        if t2:
            texts.append(f"SOURCE 2 ({url2.strip()}):\n{t2}")

    st.session_state.url_context = "\n\n---\n\n".join(texts) if texts else ""

# SUMMARY -----------------------------------
def maybe_summarize():
    """
    If there are more than 12 messages, summarize the oldest ones and keep the 
    last 6. Uses OpenAI for summarization (simple and reliable).
    """
    MAX_MSGS = 12
    KEEP_LAST = 6

    if len(st.session_state.messages) <= MAX_MSGS:
        return
    
    older = st.session_state.messages[:-KEEP_LAST]
    convo = "\n".join([f'{m["role"].upper()}: {m["content"]}' for m in older])

    resp = openai_client.chat.completions.create(
        model=st.session_state.openai_model,
        messages=[
            {"role": "system", "content": "Summarize the conversation. Keep key facts, preferences, and decisions."},
            {"role": "user", "content": convo},
        ],
        temperature=0.7
    )
    new_summary = resp.choices[0].message.content.strip()

    st.session_state.summary = (st.session_state.summary + "\n" + new_summary).strip()
    st.session_state.messages = st.session_state.messages[-KEEP_LAST:]

def build_messages_for_model():
    msgs = [{"role": "system", "content": PERSONA}]

    if st.session_state.url_context:
        msgs.append({"role": "system", "content": "Use these sources as context:\n\n" + st.session_state.url_context})

    if st.session_state.summary:
        msgs.append({"role": "system", "content": "Conversation summary so far:\n" + st.session_state.summary})

    msgs.extend(st.session_state.messages)  # buffer
    return msgs

# STREAMING ---------------------------------

def stream_openai(model_messages):
    stream = openai_client.chat.completions.create(
        model=st.session_state.openai_model,
        messages=model_messages,
        stream=True,
    )
    for event in stream:
        delta = event.choices[0].delta
        if delta and getattr(delta, "content", None):
            yield delta.content

def stream_claude(model_messages):
    # Simple approach: flatten into one prompt (keeps code legible)
    prompt = "\n\n".join([f'{m["role"].upper()}: {m["content"]}' for m in model_messages])
        
    with claude_client.messages.stream(
        model=st.session_state.claude_model,
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}],
    ) as s:
        for text in s.text_stream:
            yield text
    
# SIDEBAR -----------------------------------

with st.sidebar:
    st.header("Sources")
    url1 = st.text_input("URL 1")
    url2 = st.text_input("URL 2 (optional)")

    if st.button("Load URL(s)"):
        load_urls(url1, url2)
        st.success("URLs loaded!" if st.session_state.url_context else "No URL content loaded.")

    st.divider()
    st.header("Model")

    st.session_state.provider = st.radio("Choose LLM", ["OpenAI", "Claude"])

    if st.session_state.provider == "OpenAI":
        st.session_state.openai_model = st.selectbox("OpenAI model", ["gpt-4.0", "gpt-4.1", "gpt-4.6"])
    else:
        st.session_state.claude_model = st.selectbox("Claude model", ["claude-sonnet-4-5", "claude-opus-4.6"])

    st.divider()
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.session_state.summary = ""
        st.success("Cleared.")


# Display CHAT HISTORY ---------------------------------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Chat Input + Streaming Response -----------------------
if prompt := st.chat_input("Ask Chatty G something about your webpage."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    maybe_summarize()
    model_message = build_messages_for_model()

    with st.chat_message("assistant"):
        if st.session_state.provider == "OpenAI":
            answer = st.write_stream(stream_openai(model_message))
        else:
            answer = st.write_stream(stream_claude(model_message))

    st.session_state.messages.append({"role": "assistant", "content": answer})
