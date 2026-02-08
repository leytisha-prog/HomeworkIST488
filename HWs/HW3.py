import streamlit as st
import requests
from bs4 import BeautifulSoup 
from openai import OpenAI
import anthropic

# Name of APP and description

st.title("Chatty G")
st.write("An app that summarizes the contents of web pages.")

openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
claude_client = Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

PERSONA = "You are Chatty G, a helpful and friendly assistant."

# SESSION STATE
if "messages" not in st.session_state:
    st.session_state.messages = []      # user/assistant only (display + buffer)
if "url_context" not in st.session_state: 
    st.session_state.url_context = ""  # combined text from URL(s)
if "summary" not in st.session_state:
    st.session_state.summary = "" # compress memory

# URL reading (prof's code altered)
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

# SUMMARY 
def maybe_summarize():
    """
    If there are more than 12 messages in the chat,
    summarize the oldest ones into st.session_state.summary
    and keep only the last 6.

    """
    MAX_MGS = 12
    KEEP_LAST = 6

    if len(st.session_state.messages) <= MAX_MGS:
        return
    
    older = st.session_state.messages[:-KEEP_LAST]
    if not older:
        return
    
    convo = "\n".join([f'{m["role"].upper()}: {m["content"]}' for m in older])

    resp = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "Summarize this conversation. Keep key facts, user preferences, and decisions."},
            {"role": "user", "content": convo}
        ],
        temperature=0.2,
    )
    new_summary = resp.choices[0].message.content.strip()

    st.session_state.summary = (st.session_state.summary + "\n" + new_summary).strip()
    st.session_state.messages = st.session_state.messages[-KEEP_LAST:]

# Build messages for the model (persona + urls, summary + buffer)
def build_messages():
    msgs = [{"role": "system", "content": PERSONA}]

    if st.session_state.url_context:
        msgs.append({"role": "system", "content": "Use these sources as context:\n\n" + st.session_state.url_context})
    
    if st.session_state.summary:
        msgs.append({"role": "system", "content": "Conversation summary so far:\n" + st.session_state.summary})

    msgs.extend(st.session_state.messages)  # buffer
    return msgs