import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

# 1. Load CSV
df = pd.read_csv("news.csv")

# 2. Embedding function 
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key="OPENAI_API_KEY",
    model_name="text-embedding-3-small"
)

# 3. Create DB
client = chromadb.PersistentClient(path="./chroma_db_data")

collection = client.get_or_create_collection(
    name="news_collection",
    embedding_function=openai_ef
)

# 4. Convert rows to documents from csv 
documents = [
    f"Title: {row['title']}\nDate: {row['date']}\nArticle: {row['body_column']}"
    for _, row in df.iterrows()
]

metadatas = df.to_dict(orient="records")
ids = [str(i) for i in range(len(df))]

# 5. Store 
collection.upsert(
    documents=documents,
    metadatas=metadatas,
    ids=ids
)

print("DB built successfully!") # run this once. 

# STREAMLIT APP
import streamlit as st
import chromadb
from openai import OpenAI

# 5. Load DB
client = chromadb.PersistentClient(path="./chroma_db_data")
collection = client.get_collection(name="news.csv")

openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.title("News RAG Bot")
query = st.text_input("Ask about the news:")

if query:

    # Retrieve relevant articles 
    results = collection.query(
        query_texts=[query],
        n_results=5
    )

    docs = results["documents"][0]
    metadata = results["metadatas"][0]

    context = "\n\n".join(docs)

    # --------- MODEL 1 (Retrieval) for testing -------
    response_1 = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"""
            You are a news assistant.

            Use the articles below to answer the question.

            Articles:
            {context}

            Question:
            {query}
            """
        }]
)

answer = response_1.choices[0].message.content

st.subheader("Answer")
st.write(answer)

# ------------ MODEL 2 (Ranking/interesting news) -------
if "interesting" in query.lower():

    ranking_prompt = f"""
    Given the following news articles, rank them from interesting to least interesting.
    Define "interesting" based on:
    - impact
    - novelty
    - relevance 

    Return a numbered list with brief justification for citation purposes.

    Articles:
    {context}
    """
    response_2 = openai_client.chat.completions.create(
        model="gpt-4o", # second model
        messages=[{"role": "user", "content": ranking_prompt}]
    )

    st.subheader("Ranked News")
    st.write(response_2.choices[0].message.content)

# Show sources
st.subheader("Sources")
st.write(metadata)


