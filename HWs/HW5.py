# 1) SQLite patch for Chroma MUST BE FIRST
import sys
__import__("pysqlite3")
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

# 2) Imports
import json
import time
import random 
from pathlib import Path
import streamlit as st
from openai import OpenAI
import chromadb

# For HTML to text
from bs4 import BeautifulSoup

#------- App UI 
st.title("HW5: Short-Term Memory Chatbot and Tool-Calling RAG")
st.caption("Uses a tool function that retrieves from Chroma and calls the LLM to answer.")

# ----------------------------
# 3. Paths 
# ----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]   # repo root
HTML_FOLDER = BASE_DIR / "HWs" / "html-websites" 
CHROMA_DIR = Path("/tmp") / "ChromaDB_for_HW5"      
COLLECTION_NAME = "HW5Collection"
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4.1-mini"

# -----------------------------
# OpenAI client
# ------------------------------
if "openai_client" not in st.session_state:
    st.session_state.openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# -----------------------------
# Chroma collection 
# -----------------------------
@st.cache_resource
def get_collection():
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return chroma_client.get_or_create_collection(COLLECTION_NAME)

collection = get_collection()
st.write("Chunks in collection:", collection.count())

# -----------------------------
# Short-term memory: last 5 interactions = last 10 messages
# -----------------------------
if "hw5_messages" not in st.session_state:
    st.session_state.hw5_messages = []

def last_5_turns(messages):
    return messages[-10:]

# -----------------------------
# Retrieval helper (no LLM call here)
# -----------------------------
def retrieve_context(query: str, k: int = 5):
    client = st.session_state.openai_client
    q_emb = client.embeddings.create(input=query, model=EMBED_MODEL).data[0].embedding

    results = collection.query(
        query_embeddings=[q_emb],
        n_results=k,
        include=["documents", "metadatas"],
    )

    docs = (results.get("documents") or [[]])[0]
    metas = (results.get("metadatas") or [[]])[0]
    ids = (results.get("ids") or [[]])[0]

    blocks, sources = [], []
    for doc, meta, doc_id in zip(docs, metas, ids):
        meta = meta or {}
        src = meta.get("source", doc_id)
        ch = meta.get("chunk", "")
        label = f"{src} (chunk {ch})" if ch != "" else src
        sources.append(label)
        blocks.append(f"[SOURCE: {label}]\n{doc}")

    context = "\n\n---\n\n".join(blocks)[:12000]
    return context, sources

# -----------------------------
# REQUIRED TOOL FUNCTION:
# - takes query
# - does vector search
# - calls LLM with retrieved excerpts (NO tools here)
# -----------------------------
def relevant_club_info(query: str) -> dict:
    context, sources = retrieve_context(query, k=5)

    client = st.session_state.openai_client
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant for questions about student organizations. "
                "Use the retrieved organization webpage excerpts when relevant. "
                "If the excerpts do not contain the answer, say so and answer generally."
            ),
        },
        {
            "role": "user",
            "content": f"""
USER QUESTION:
{query}

RETRIEVED ORG WEBPAGE EXCERPTS:
{context}

Instructions:
- If you used the excerpts, say: "Used org webpages via RAG."
- If you did NOT use them, say: "Not found in webpages; answering generally."
- Be concise, but helpful. If you mention an organization, name it clearly.
""".strip(),
        },
    ]

    # IMPORTANT: no tools passed here → prevents calling itself again
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.3,
    )

    answer = resp.choices[0].message.content
    return {"answer": answer, "sources": sources}

# -----------------------------
# Tool schema exposed to model
# -----------------------------
tools = [
    {
        "type": "function",
        "function": {
            "name": "relevant_club_info",
            "description": "Retrieve relevant student organization webpage info and return an answer using those excerpts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The user question to search for."}
                },
                "required": ["query"],
            },
        },
    }
]

def run_tool(tool_name: str, tool_args: dict) -> dict:
    if tool_name == "relevant_club_info":
        return relevant_club_info(tool_args["query"])
    return {"answer": "Unknown tool.", "sources": []}

# -----------------------------
# Chat UI
# -----------------------------
for m in st.session_state.hw5_messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

user_q = st.chat_input("Ask about student organizations...")

if user_q:
    st.session_state.hw5_messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.write(user_q)

    memory = last_5_turns(st.session_state.hw5_messages)

    # 1) First LLM call: allow tool-calling
    resp1 = st.session_state.openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. "
                    "When questions relate to student organizations, call the relevant_club_info tool."
                ),
            },
            *memory,
        ],
        tools=tools,
        tool_choice="auto",
        temperature=0.3,
    )

    msg1 = resp1.choices[0].message

    # 2) If tool was called, execute it and display result
    if msg1.tool_calls:
        tool_call = msg1.tool_calls[0]
        args = json.loads(tool_call.function.arguments)

        tool_result = run_tool(tool_call.function.name, args)

        assistant_text = tool_result["answer"]
        sources = tool_result.get("sources", [])

        st.session_state.hw5_messages.append({"role": "assistant", "content": assistant_text})
        with st.chat_message("assistant"):
            st.write(assistant_text)
            if sources:
                st.caption("Sources: " + ", ".join(sources))
    else:
        # If model answered without using tool
        assistant_text = msg1.content or ""
        st.session_state.hw5_messages.append({"role": "assistant", "content": assistant_text})
        with st.chat_message("assistant"):
            st.write(assistant_text)
