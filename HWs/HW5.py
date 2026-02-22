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
    chroma_client = chromadb.PersistentClient(path="./ChromaDB_for_HW5")
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

# ----------------------------
# 5. Helpers: HTML -> text 
# ----------------------------
def html_to_text(html_str: str) -> str: 
    """Convert HTML -> plain text for embedding. We strip scripts/styles and collapse whitespace.""" 
    
    soup = BeautifulSoup(html_str, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n") # preserve some structure with newlines

    # Collapse multiple newlines and whitespace
    lines = [ln.strip() for ln in text.splitlines()] 
    lines = [ln for ln in lines if ln] # remove empty lines
    return "\n".join(lines)

   
def chunk_into_two(text: str) -> tuple[str, str]:
    """REQUIRED: creat exactly TWO chunks per document. 
    Chunking method:
    - We split the text into two halves by character count,
    then "snap" the split to the nearest newline to avoid cutting sentences midline.
    Why this method:
    - Simple, deterministic, and meets the assignment requirements ("two mini documents per file")
    - Keeps chunks roughly equal size, which helps retrieval coverage."""
    
    text = text.strip()
    if len(text) <10:
        return text, "" # short docs go in chunk 1, leave chunk 2 empty
    mid = len(text) // 2

    # Snap to nearest newline
    left_nl = text.rfind("\n", 0, mid)
    right_nl = text.find("\n", mid)

    if left_nl == 1 and right_nl == 1:
        split_at = mid
    elif left_nl == -1:
        split_at = right_nl
    elif right_nl == -1:
        split_at = left_nl
    else:
        # Choose the split that results in more balanced chunk sizes
        split_at = left_nl if (mid - left_nl) <= (right_nl - mid) else right_nl
    
    chunk1 = text[:split_at].strip()
    chunk2 = text[split_at:].strip()
    return chunk1, chunk2

def embed_with_retry(text: str, max_retries: int = 5) -> list:
    """
    Embeddings can sometimes throw transient 500 errors.
    Retry with exponential backoff + jitter. 
    Also cap length for stability (embedding models have limits, and I have had issues with very long documents).
    """

    client = st.session_state.openai_client
    text = (text or "") [:12000] # cap length to avoid issues with very long documents

    
    for attempt in range(1, max_retries + 1):
        try:
            return client.embeddings.create(input=text, model=EMBED_MODEL).data[0].embedding
        except Exception as e:
            if attempt == max_retries:
                raise
            time.sleep((2 ** attempt) + random.random())


# ----------------------------
# Vector DB Builder (ONLY ONCE)
# ----------------------------

def build_hw4_vectordb():
    """
    Build the Chroma vector DB from HTML files.
    IMPORTANT: Only build if not already populated (saves time + cost).
    Stores two chunks per HTML file. 
    """
    
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = chroma_client.get_or_create_collection(COLLECTION_NAME)


    # If already built, return immediately (saves cost)
    if collection.count() > 0:
        return collection

    html_files = sorted(HTML_FOLDER.glob("*.html"))

    progress = st.progress(0)
    status = st.empty()
    added = 0

    for i, html_file in enumerate(html_files, start=1):
        status.write(f"Indexing {i}/{len(html_files)}: {html_file.name}")

        raw_html = html_file.read_text(encoding="utf-8", errors="ignore")
        text = html_to_text(raw_html)

        # Create exactly two chunks per document
        c1, c2 = chunk_into_two(text)

        # Add Chunk 1
        if c1:
            try:
                emb1 = embed_with_retry(c1)
                collection.add(
                    documents=[c1[:12000]],  # cap length for stability
                    ids=[f"{html_file.stem}_chunk1"],
                    embeddings=[emb1],
                    metadatas=[{"source": html_file.name, "chunk": 1}],
                    
                )
                added += 1
            except Exception as e:
                st.error(f"Failed embedding {html_file.name} chunk1: {e}")

        # Add Chunk 2
        if c2:
            try:
                emb2 = embed_with_retry(c2)
                collection.add(
                    documents=[c2[:12000]], # cap length for stability
                    ids=[f"{html_file.stem}_chunk2"],
                    embeddings=[emb2],
                    metadatas=[{"source": html_file.name, "chunk": 2}],

                )
                added += 1
            except Exception as e:
                st.error(f"Failed embedding {html_file.name} chunk2: {e}")  

        progress.progress(i / max(1, len(html_files)))    

    status.write("Indexing complete.")
    st.success(f"Loaded {added} chunks from HTML files into ChromaDB.")
    return collection


# ----------------------------
# Build/reuse vector DB in session_state (assignment requirement)
# ----------------------------

if "HW4_VectorDB" not in st.session_state:
    with st.spinner("Building HW4 vector DB (first run only)..."):
        st.session_state.HW4_VectorDB = build_hw4_vectordb()

collection = st.session_state.HW4_VectorDB
st.write("Chunks in collection:", collection.count())

# Note: The above code builds the vector DB and stores it in session_state.


# -----------------------------
# Maintenance: rebuild button (optional, but helpful)
# -----------------------------
st.sidebar.header("Maintenance")
if st.sidebar.button("Delete & Rebuild Vector DB"):
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    st.session_state.pop("Lab5_VectorDB", None)
    st.success("Deleted collection. Refresh page to rebuild.")

# ----------------------------
# Retrieval + Answering 
# ----------------------------
def retrieve_context(question: str, k: int = 5) -> tuple[str, list[str]]:
    q_emb = embed_with_retry(question)

    results = collection.query(
        query_embeddings=[q_emb],
        n_results=k,
        include=["documents", "metadatas"],
    )

    docs = (results.get("documents") or [[]])[0] 
    metas = (results.get("metadatas") or [[]])[0]   

    blocks = []
    sources = []
    for i, doc in enumerate(docs):
        meta = metas[i] if i < len(metas) and metas[i] else {}
        src = meta.get("source", f"doc_{i+1}")
        ch = meta.get("chunk", "?")
        sources.append(f"{src} (chunk {ch})")
        blocks.append(f"[SOURCE: {src} | chunk {ch}]\n{doc}")

    context = "\n\n---\n\n".join(blocks)[:12000]
    return context, sources 

def answer_with_rag_and_memory(question: str, context: str, memory_messages: list[dict]) -> str:
    """
    Memory buffer requirement:
    - Keep only last 5 interactions (turns).
    We pass those last turns and retrieved context and current question to the LLM.
    """
    client = st.session_state.openai_client

    system = {
        "role": "system",
        "content": (
            "You are a helpful chatbot that answers questions about student organizations."
            "You are given retrieved organization webpage excerpts via RAG."
            "Use the excerpts when relevant, and cite which page/chunk you used."
            "If the answer is not in the excerpts, answer using general knowledge and say it is not from the provided webpages."
        )       
    }   

    user_with_context = {
        "role": "user",
        "content": f"""
QUESTION:
{question}

RETRIEVED WEBPAGE EXCERPTS (RAG):
{context}

Instructions:
- If the excerpts contain relevant information, use them and cite pages/chunks.
- If not, answer generally and say you did not use the excerpts.
""".strip() 
    }   

    messages = [system] + memory_messages + [user_with_context]

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.3
    )
    return resp.choices[0].message.content

# ----------------------------
# Chat UI + Memory Buffer (last 5 interactions)
# ----------------------------

st.header("Ask a question about student organizations")

if "hw4_memory" not in st.session_state:
    st.session_state.hw4_memory = [] # full chat history for display

# Display chat history
for m in st.session_state.hw4_memory:
    with st.chat_message(m["role"]):
        st.write(m["content"])

user_q = st.chat_input("Type your question here...")

if user_q:
    # Append user message
    st.session_state.hw4_memory.append({"role": "user", "content": user_q}) 
    with st.chat_message("user"):
        st.write(user_q)

    # Build memory buffer: last 5 interactions = last 10 messages (user+assistant),
    # but only from the *end* of the conversation.
    # (We keep the display history bigger, but only feed the last 5 turns to the LLM.)
    last_10 = st.session_state.hw4_memory[-10:]
    memory_for_llm = [{"role": m["role"], "content": m["content"]} for m in last_10]

    with st.spinner("Retrieving relevant org pages + generating answer..."):
        context, sources = retrieve_context(user_q, k=5)
        answer = answer_with_rag_and_memory(user_q, context, memory_for_llm)

    st.session_state.hw4_memory.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.write(answer)
        if sources:
            st.caption("Sources: " + ", ".join(sources))


# ----------------------------------------------------------------------- 
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
