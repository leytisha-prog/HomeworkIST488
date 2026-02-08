import streamlit as st
import requests
from bs4 import BeautifulSoup 
from openai import OpenAI
import anthropic

# Name of APP and description

st.title("Chatty G")
st.write("An app that summarizes the contents of web pages.")

PERSONA = "You are Chatty G, a helpful and friendly assistant."

# SESSION STATE ------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []      # user/assistant only (display + buffer)
if "url_context" not in st.session_state: 
    st.session_state.url_context = ""  # combined text from URL(s)
if "summary" not in st.session_state:
    st.session_state.summary = "" # compress memory

# Client - secret keys in streamlit
openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
claude_client = Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])


# SESSION DEFAULTS with LLMs advanced models -------------------
if "provider" not in st.session_state:
    st.session_state.provider = "OpenAI"

if "openai_model" not in st.session_state:
    st.session_state.openai_model = "gpt-4o"  

if "claude_model" not in st.session_state:
    st.session_state.claude_model = "claude-3-5-sonnet-latest"  

# SIDEBAR -----------------------------------

with st.sidebar:
    st.header("Sources")
    # (your URL inputs here)

    st.divider()
    st.header("Model")

    st.session_state.provider = st.radio(
        "Choose LLM",
        ["OpenAI", "Claude"]
    )

    if st.session_state.provider == "OpenAI":
        st.session_state.openai_model = st.selectbox(
            "OpenAI model",
            ["gpt-4o", "gpt-4.1", "gpt-4o-mini"]
        )
    else:
        st.session_state.claude_model = st.selectbox(
            "Claude model",
            ["claude-3-5-sonnet-latest", "claude-3-5-haiku-latest"]
        )
# SIDEBAR - URLS ------------------------------------------
with st.sidebar:
    st.header("Sources")
    url1 = st.text_input("URL 1")
    url2 = st.text_input("URL 2 (optional)")

    if st.button("Load URL(s)"):
        load_urls(url1, url2)
        st.success("URLs loaded!" if st.session_state.url_context else "No URL content loaded.")

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.session_state.summary = ""
        st.success("Cleared.")


# URL reading (PROF'S CODE ALTERED) ------------------------
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

# SUMMARY ---------------------------------------
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

# BUILD MESSAGES FOR MODELS (persona + urls, summary + buffer)
messages_for_model = build_messages()  # however you assemble persona + urls + summary + buffer

with st.chat_message("assistant"):
    if st.session_state.provider == "OpenAI":
        stream = openai_client.chat.completions.create(
            model=st.session_state.openai_model,
            messages=messages_for_model,
            stream=True,
        )
        # wrap to yield text chunks
        def openai_text(stream):
            for event in stream:
                delta = event.choices[0].delta
                if delta and getattr(delta, "content", None):
                    yield delta.content

        answer = st.write_stream(openai_text(stream))

    else:
        # Claude streaming
        def claude_text():
            # Convert OpenAI-style messages to a single prompt (simple approach)
            # (Keeps your app readable; you can improve later)
            prompt = "\n\n".join([f'{m["role"].upper()}: {m["content"]}' for m in messages_for_model])

            with claude_client.messages.stream(
                model=st.session_state.claude_model,
                max_tokens=800,
                messages=[{"role": "user", "content": prompt}],
            ) as s:
                for text in s.text_stream:
                    yield text

        answer = st.write_stream(claude_text())

st.session_state.messages.append({"role": "assistant", "content": answer})

