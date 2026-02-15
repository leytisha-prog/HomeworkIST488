# 1) SQLite patch for Chroma MUST BE FIRST
import sys
__import__("pysqlite3")
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

# 2) Imports
import time
import random
import streamlit as st
from openai import OpenAI
import chromadb
from pathlib import Path
from PyPDF2 import PdfReader


# ----------------------------
# 3. Paths (PDFs are inside Labs/Lab-04-Data)
# ----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]             # repo root
PDF_FOLDER = BASE_DIR / "Labs" / "Lab-04-Data"             # 7 PDFs live here

# On Streamlit Cloud, /tmp is the safest writable location
CHROMA_DIR = Path("/tmp") / "ChromaDB_for_Lab"

COLLECTION_NAME = "Lab4Collection"
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4.1-mini"  #  gpt-5-mini doesn't appear to work with the temperature0.3


# ----------------------------
# 4. App UI
# ----------------------------
st.title("Lab 4: Course Information Chatbot (RAG)")

# Helpful sanity check while developing (I can remove later)
st.caption(f"PDF folder: {PDF_FOLDER}")
st.caption(f"PDFs found: {[p.name for p in PDF_FOLDER.glob('*.pdf')]}")

# OpenAI client
if "openai_client" not in st.session_state:
    st.session_state.openai_client = OpenAI(api_key=st.secrets["OPEN_AI_KEY"])


# ----------------------------
# 5. Helpers: PDF -> text, embedding with retry
# ----------------------------
def extract_text_from_pdf(pdf_path: str, max_pages: int = 6) -> str:
    """Read PDF and return text (cap pages to keep embedding fast)."""
    reader = PdfReader(pdf_path)
    chunks = []
    for page in reader.pages[:max_pages]:
        txt = page.extract_text()
        if txt:
            chunks.append(txt)
    return "\n".join(chunks).strip()


def embed_with_retry(text: str, max_retries: int = 5) -> list:
    """Retry transient OpenAI 500 errors; cap text to keep requests stable."""
    client = st.session_state.openai_client
    text = (text or "")[:12000]

    for attempt in range(1, max_retries + 1):
        try:
            return client.embeddings.create(input=text, model=EMBED_MODEL).data[0].embedding
        except Exception as e:
            if attempt == max_retries:
                raise
            time.sleep((2 ** attempt) + random.random())


# ----------------------------
# PART A: REQUIRED builder function
# ----------------------------
def build_lab4_vectordb():
    """
    REQUIRED by assignment:
    - Construct ChromaDB collection named 'Lab4Collection'
    - Use OpenAI embeddings
    - Read 7 PDFs into text
    - Use filename as key
    - Use metadata as needed
    - Return the collection
    """
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

    # If already built, return immediately (saves cost)
    if collection.count() > 0:
        return collection

    pdf_files = sorted(PDF_FOLDER.glob("*.pdf"))

    progress = st.progress(0)
    status = st.empty()
    added = 0

    for i, pdf_file in enumerate(pdf_files, start=1):
        status.write(f"Indexing {i}/{len(pdf_files)}: {pdf_file.name}")

        text = extract_text_from_pdf(str(pdf_file))
        if not text:
            st.warning(f"Skipping {pdf_file.name} (no extractable text).")
            progress.progress(i / max(1, len(pdf_files)))
            continue

        doc_id = pdf_file.stem.replace(" ", "_")  # key = filename (safe)
        try:
            emb = embed_with_retry(text)
            collection.add(
                documents=[text[:12000]],
                ids=[doc_id],
                embeddings=[emb],
                metadatas=[{"source": pdf_file.name}],
            )
            added += 1
        except Exception as e:
            # Continue so I don't end up with 0 PDFs loaded
            st.error(f"Failed to embed {pdf_file.name}: {e}")
            progress.progress(i / max(1, len(pdf_files)))
            continue

        progress.progress(i / max(1, len(pdf_files)))

    status.write("Indexing complete.")
    st.success(f"Loaded {added} PDFs into ChromaDB.")
    return collection


# ----------------------------
# Store VectorDB in session_state ONLY ONCE (assignment requirement)
# ----------------------------
if "Lab4_VectorDB" not in st.session_state:
    # This is the only place the build function is called automatically
    with st.spinner("Building Lab4 vector DB (first run only)..."):
        st.session_state.Lab4_VectorDB = build_lab4_vectordb()

collection = st.session_state.Lab4_VectorDB
st.write("Docs in collection:", collection.count())


# ----------------------------
# PART A Test (remove later per Chris' instructions)
# ----------------------------
#st.subheader("Part A: VectorDB Test (remove for final submission)")
#test_query = st.text_input("Test search string", value="Generative AI")
#if st.button("Run test search"):
    #q_emb = embed_with_retry(test_query)
    #results = collection.query(
        #query_embeddings=[q_emb],
        #n_results=3,
        #include=["metadatas"]   # don't include ids to avoid Chroma version errors -- I have been having issues with this. 
   #)
    #metas = (results.get("metadatas") or [[]])[0]
    #top_files = []
    #for meta in metas:
        #meta = meta or {}
        #top_files.append(meta.get("source", "unknown"))

    #st.write("Top 3 returned documents:")
    #for i, f in enumerate(top_files, start=1):
        #st.write(f"{i}. {f}")


# ----------------------------
# PART B: Chatbot using RAG
# ----------------------------
st.divider()
st.subheader("Part B: Course Information Chatbot (RAG)")

def retrieve_context(question: str, k: int = 7):
    q_emb = embed_with_retry(question)
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=k,
        include=["documents", "metadatas"]
    )
    docs = (results.get("documents") or [[]])[0]
    metas = (results.get("metadatas") or [[]])[0]

    blocks = []
    sources = []
    for i, doc in enumerate(docs):
        meta = metas[i] if i < len(metas) and metas[i] else {}
        src = meta.get("source", f"doc_{i+1}")
        sources.append(src)
        blocks.append(f"[SOURCE: {src}]\n{doc}")

    context = "\n\n---\n\n".join(blocks)[:12000]
    return context, sources


def answer_with_rag(question: str, context: str, sources: list[str]) -> str:
    client = st.session_state.openai_client

    messages = [
        {
            "role": "system",
            "content": (
                "You are a course information chatbot. "
                "You are given syllabus excerpts retrieved via RAG. "
                "If the answer is supported by the excerpts, use them and clearly say you used course documents via RAG. "
                "If the answer is not supported by the excerpts, answer using general knowledge and clearly say it is not from the course documents."
            )
        },
        {
            "role": "user",
            "content": f"""
QUESTION:
{question}

SYLLABUS EXCERPTS (RAG):
{context}
""".strip()
        }
    ]

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.3
    )
    return resp.choices[0].message.content


# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

user_q = st.chat_input("Ask a question about the syllabi...")

if user_q:
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.write(user_q)

    with st.spinner("Retrieving syllabi + generating answer..."):
        context, sources = retrieve_context(user_q, k=7)
        answer = answer_with_rag(user_q, context, sources)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.write(answer)
        st.caption("Sources: " + ", ".join(sources))


# ----------------------------
# Optional maintenance  - helps with rebuilding vectorDB
# ----------------------------
st.sidebar.header("Maintenance (optional)")
if st.sidebar.button("Delete collection and rebuild"):
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    st.session_state.pop("Lab4_VectorDB", None)
    st.success("Deleted. Refresh page to rebuild.")