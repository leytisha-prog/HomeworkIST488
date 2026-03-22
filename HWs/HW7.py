import pandas as pd 
import streamlit as st

#df = pd.read_csv(
    #"HWs/news.csv",
    #sep=";",
    #engine="python",
    #quotechar='"',
    #on_bad_lines="skip"
#)
# Above code is built and RAN ONCE - The code below is later added AFTER
# the file build_db.py is built
# ---------------------------------

import chromadb
from openai import OpenAI

client = chromadb.PersistentClient(path="./chroma_db_data")
collection = client.get_collection(name="news_collection")

openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

query = st.text_input("Ask about the news:")

if query:
    results = collection.query(query_texts=[query], n_results=5)
    docs = results["documents"][0]

    context = "\n\n".join(docs)

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

    st.write(response.choices[0].message.content)


if "interesting" in query.lower():
    ranking_prompt = f"""
    Rank these news articles from most interesting to least interesting.

    Criteria:
    - impact
    - novelty
    - relevance

    Articles:
    {context}
    """

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": ranking_prompt}]
    )

    st.write(response.choices[0].message.content)


