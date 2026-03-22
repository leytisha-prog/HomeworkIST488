
# SQLite fix for Chroma (MUST be first)
import sys
__import__("pysqlite3")
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")


import pandas as pd 
import streamlit as st

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

import chromadb
from openai import OpenAI
import os

# ----------------------------
# App UI
# ----------------------------
st.title("HW7: Get Your News!")
st.caption("CSV-Based RAG")

model_choice = st.selectbox(
    "Choose a model:",
    ["gpt-4o-mini", "gpt-4o"]
)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

client = chromadb.PersistentClient(
    path=os.path.join(BASE_DIR, "chroma_db_data")
)

#client = chromadb.PersistentClient(path="./chroma_db_data")
collection = client.get_collection(name="news_collection")


openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

query = st.text_input("Ask about the news:")

st.subheader("Articles Being Ranked")

if query:
    # 1. Retrieve from Chroma
    results = collection.query(
        query_texts=[query],
        n_results=5
    )

    docs = results["documents"][0]

    # 2. Build context (THIS is what you're missing)
    context = "\n\n".join(docs)

    # 3. Normal answer
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"""
            Use the articles below to answer the question.

            {context}

            Question: {query}
            """
        }]
    )

    st.subheader("Answer")
    st.write(response.choices[0].message.content)

    # 4. 🔥 Ranking comparison (NOW context exists)
    if "interesting" in query.lower():

        ranking_prompt = f"""
        Rank the following news articles from MOST interesting to LEAST interesting.

        Articles:
        {context}
        """

        mini_response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": ranking_prompt}]
        )

        full_response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": ranking_prompt}]
        )

        st.subheader("Ranking Comparison")

        col1, col2 = st.columns(2)

        with col1:
            st.write("Mini Model")
            st.write(mini_response.choices[0].message.content)

        with col2:
            st.write("Full Model")
            st.write(full_response.choices[0].message.content)


