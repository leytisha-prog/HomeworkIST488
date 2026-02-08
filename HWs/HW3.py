from openai import OpenAI
import streamlit as st
import tiktoken


st.title ("Chatty G - Lab 3: Streamlit Chat Interface")

# Below is the code to set up OpenAI client and default model - pull responses from secrets

# Set OpenAI API key from Streamlit secrets
client = OpenAI(api_key=st.secrets["OPEN_AI_KEY"])

# Set a default model and max tokens for the chat completions
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4-turbo"


# Below is the code for a simple chat interface using Streamlit's chat components
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system", 
            "content": (
                "You are Chatty G, a helpful and friendly assistant."
                "Explain the URL shortening process in simple terms, suitable for a 10-year-old."
                "Summarize the URL in simple terms, suitable for a 10-year-old." 
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
   



    





