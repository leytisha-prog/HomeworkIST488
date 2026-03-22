
# SQLite fix for Chroma (MUST be first)
#import sys
#__import__("pysqlite3")
#sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")


#import pandas as pd 
#import streamlit as st

#df = pd.read_csv(
    #"HWs/news.csv",
    #sep=";",
    #engine="python",
    #quotechar='"',
    #on_bad_lines="skip"
#) - Code above ALSO addresses csv parsing errors 
# Above code is built and RAN ONCE - The code below is later added AFTER
# the file build_db.py is built
# ---------------------------------

# -------------------------------
# 1. SQLite fix (MUST be first)
# -------------------------------
import sys
__import__("pysqlite3")
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

# -------------------------------
# 2. Imports
# -------------------------------
import streamlit as st
import chromadb
from openai import OpenAI
import os

# -------------------------------
# 3. Setup paths + DB
# -------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

client = chromadb.PersistentClient(
    path=os.path.join(BASE_DIR, "chroma_db_data")
)

collection = client.get_collection(name="news_collection")


# OpenAI client
openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# -------------------------------
# 4. UI
# -------------------------------
st.write("Collection count:", collection.count())
st.title("HW7: LawFirst Client News Bot! 📊")
st.caption("CSV-Based RAG with Ranking + Model Comparison")

model_choice = st.selectbox(
    "Choose a model for answering:",
    ["gpt-4o-mini", "gpt-4o"]
)

query = st.text_input("Ask about the news:", key="news_query")

# -------------------------------
# 5. Main RAG Logic
# -------------------------------
if query:

    # Retrieve relevant docs
    results = collection.query(
        query_texts=[query],
        n_results=10
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    if not docs:
        st.warning("No relevant articles found.")
    else:
        # -------------------------------
        # 6. Structured context (IMPORTANT)
        # -------------------------------
        structured_context = ""

        for i, (doc, meta) in enumerate(zip(docs, metas)):
            structured_context += f"""
            Article {i+1}:
            Company: {meta.get("company", "")}
            Date: {meta.get("date", "")}
            Content: {doc}
            --------------------
            """

        # Show retrieved articles
        st.subheader("📄 Articles Retrieved")
        st.write(docs)

        # -------------------------------
        # 7. Answer generation
        # -------------------------------
        answer_prompt = f"""
        You are a financial news assistant.

        Use ONLY the articles below to answer the question.

        Articles:
        {structured_context}

        Question:
        {query}
        """

        response = openai_client.chat.completions.create(
            model=model_choice,
            messages=[{"role": "user", "content": answer_prompt}]
        )

        st.subheader("Answer")
        st.write(response.choices[0].message.content)

        # -------------------------------
        # 8. Ranking (if "interesting")
        # -------------------------------
        if "interesting" in query.lower():

            ranking_prompt = f"""
            You are a news analyst.

            Rank the following articles from MOST interesting to LEAST interesting.

            Criteria:
            - impact (economic, societal, or technological)
            - novelty (new or surprising developments)
            - relevance (importance to current events)

            Instructions:
            - Refer to articles by number (Article 1, Article 2, etc.)
            - Provide ranking with explanation

            Articles:
            {structured_context}
            """

            # Model 1 (cheap)
            mini_response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": ranking_prompt}]
            )

            # Model 2 (strong)
            full_response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": ranking_prompt}]
            )

            # Display comparison
            st.subheader("Ranking Comparison")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("GPT-4o-mini")
                st.write(mini_response.choices[0].message.content)

            with col2:
                st.markdown("GPT-4o")
                st.write(full_response.choices[0].message.content)

        # -------------------------------
        # 9. Show sources
        # -------------------------------
        st.subheader("📚 Sources")
        st.write(metas)

        print("BUILD DB PATH:", os.path.abspath(os.path.join(BASE_DIR, "chroma_db_data")))