# 1) SQLite patch for Chroma MUST BE FIRST (Streamlit Cloud fix)
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
from bs4 import BeautifulSoup

# ----------------------------
# App UI
# ----------------------------
st.title("HW5: Short-Term Memory Chatbot + Tool-Calling RAG (Student Orgs)")
st.caption("Tool-calling RAG over HTML webpages + short-term memory (last 5 interactions).")

# ----------------------------
# Paths / Config
# ----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]   # repo root
HTML_FOLDER = BASE_DIR / "HWs" / "html-websites"

# Streamlit Cloud path
CHROMA_DIR = Path("/tmp") / "ChromaDB_for_HW5"

COLLECTION_NAME = "HW5Collection"
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4.1-mini"

# -----------------------------
# OpenAI client (store once)
# -----------------------------
if "openai_client" not in st.session_state:
    st.session_state.openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ----------------------------
# Helpers: HTML -> text
# ----------------------------
def html_to_text(html_str: str) -> str:
    """
    Convert HTML to plain text for embedding.
    Removes scripts/styles and collapses whitespace.
    """
    soup = BeautifulSoup(html_str, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)

def chunk_into_two(text: str) -> tuple[str, str]:
    """
    REQUIRED: create exactly TWO chunks per document.

    Chunking method:
    - Split the text into two halves by character count, then "snap" the split
      to the nearest newline so we don't cut in the middle of a line.

    Why this method:
    - Simple, deterministic, meets the requirement (two mini-documents per file)
    - Keeps chunks roughly equal size, improving retrieval coverage.
    """
    text = (text or "").strip()
    if len(text) < 10:
        return text, ""  # tiny docs: chunk2 empty but still "two chunk slots"

    mid = len(text) // 2

    left_nl = text.rfind("\n", 0, mid)
    right_nl = text.find("\n", mid)

    if left_nl == -1 and right_nl == -1:
        split_at = mid
    elif left_nl == -1:
        split_at = right_nl
    elif right_nl == -1:
        split_at = left_nl
    else:
        split_at = left_nl if (mid - left_nl) <= (right_nl - mid) else right_nl

    c1 = text[:split_at].strip()
    c2 = text[split_at:].strip()
    return c1, c2

def embed_with_retry(text: str, max_retries: int = 5) -> list:
    """
    Embeddings can throw transient errors (e.g., 500).
    Retry with exponential backoff + jitter.
    Also cap text length for stability.
    """
    client = st.session_state.openai_client
    text = (text or "")[:12000]

    for attempt in range(1, max_retries + 1):
        try:
            return client.embeddings.create(input=text, model=EMBED_MODEL).data[0].embedding
        except Exception:
            if attempt == max_retries:
                raise
            time.sleep((2 ** attempt) + random.random())

# ----------------------------
# Chroma: get collection
# ----------------------------
@st.cache_resource
def get_collection():
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return chroma_client.get_or_create_collection(COLLECTION_NAME)

# ----------------------------
# Build Vector DB ONLY if empty (assignment requirement)
# Store in st.session_state.HW5_VectorDB
# ----------------------------
def build_hw5_vectordb():
    collection = get_collection()

    if collection.count() > 0:
        return collection

    html_files = sorted(HTML_FOLDER.glob("*.html"))
    if not html_files:
        st.error(f"No HTML files found in: {HTML_FOLDER}")
        return collection

    progress = st.progress(0)
    status = st.empty()
    added = 0

    for i, html_file in enumerate(html_files, start=1):
        status.write(f"Indexing {i}/{len(html_files)}: {html_file.name}")

        raw_html = html_file.read_text(encoding="utf-8", errors="ignore")
        text = html_to_text(raw_html)
        c1, c2 = chunk_into_two(text)

        # Chunk 1
        if c1:
            try:
                emb1 = embed_with_retry(c1)
                collection.add(
                    documents=[c1[:12000]],
                    embeddings=[emb1],
                    ids=[f"{html_file.stem}_chunk1"],
                    metadatas=[{"source": html_file.name, "chunk": 1}],
                )
                added += 1
            except Exception as e:
                st.error(f"Failed embedding {html_file.name} chunk1: {e}")

        # Chunk 2
        if c2:
            try:
                emb2 = embed_with_retry(c2)
                collection.add(
                    documents=[c2[:12000]],
                    embeddings=[emb2],
                    ids=[f"{html_file.stem}_chunk2"],
                    metadatas=[{"source": html_file.name, "chunk": 2}],
                )
                added += 1
            except Exception as e:
                st.error(f"Failed embedding {html_file.name} chunk2: {e}")

        progress.progress(i / max(1, len(html_files)))

    status.write("Indexing complete.")
    st.success(f"Loaded {added} chunks into ChromaDB.")
    return collection

# Build/reuse once per session
if "HW5_VectorDB" not in st.session_state:
    with st.spinner("Building HW5 vector DB (first run only)..."):
        st.session_state.HW5_VectorDB = build_hw5_vectordb()

collection = st.session_state.HW5_VectorDB
st.write("Chunks in collection:", collection.count())

# -----------------------------
# Maintenance (optional)
# -----------------------------
st.sidebar.header("Maintenance")
if st.sidebar.button("Delete & Rebuild Vector DB"):
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    st.session_state.pop("HW5_VectorDB", None)
    st.success("Deleted collection. Refresh to rebuild.")

# ----------------------------
# Retrieval helper (ONE version only)
# ----------------------------
def retrieve_context(query: str, k: int = 5) -> tuple[str, list[str]]:
    q_emb = embed_with_retry(query)

    results = collection.query(
        query_embeddings=[q_emb],
        n_results=k,
        include=["documents", "metadatas"],
    )

    docs = (results.get("documents") or [[]])[0]
    metas = (results.get("metadatas") or [[]])[0]

    blocks, sources = [], []
    for i, doc in enumerate(docs):
        meta = metas[i] if i < len(metas) and metas[i] else {}
        src = meta.get("source", f"doc_{i+1}")
        ch = meta.get("chunk", "?")
        sources.append(f"{src} (chunk {ch})")
        blocks.append(f"[SOURCE: {src} | chunk {ch}]\n{doc}")

    context = "\n\n---\n\n".join(blocks)[:12000]
    return context, sources

# ----------------------------
# Tool function: retrieval + LLM call (NO tools inside)
# ----------------------------
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
- Be concise but helpful. If you mention an organization, name it clearly.
""".strip(),
        },
    ]

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.3,
    )

    answer = resp.choices[0].message.content
    return {"answer": answer, "sources": sources}

# ----------------------------
# Tool schema
# ----------------------------
tools = [
    {
        "type": "function",
        "function": {
            "name": "relevant_club_info",
            "description": "Retrieve relevant student organization webpage info and return an answer using those excerpts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The user's question to search for."}
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

# ----------------------------
# Short-term memory chatbot (last 5 interactions)
# ----------------------------
st.header("Ask a question about student organizations")

if "hw5_messages" not in st.session_state:
    st.session_state.hw5_messages = []

def last_5_interactions(messages: list[dict]) -> list[dict]:
    # 5 interactions = 10 messages (user+assistant)
    return messages[-10:]

# Display chat history
for m in st.session_state.hw5_messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

user_q = st.chat_input("Type your question here...")

if user_q:
    st.session_state.hw5_messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.write(user_q)

    memory = last_5_interactions(st.session_state.hw5_messages)

    # Call #1: allow tool-calling
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
        temperature=0.2,
    )

    msg1 = resp1.choices[0].message

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
        assistant_text = msg1.content or ""
        st.session_state.hw5_messages.append({"role": "assistant", "content": assistant_text})
        with st.chat_message("assistant"):
            st.write(assistant_text)