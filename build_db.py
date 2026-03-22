import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

df = pd.read_csv("HWs/news.csv", sep=";", engine="python", on_bad_lines="skip")

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key="OPENAI_API_KEY",
    model_name="text-embedding-3-small"
)

client = chromadb.PersistentClient(path="./chroma_db_data")

collection = client.get_or_create_collection(
    name="news_collection",
    embedding_function=openai_ef
)

documents = [
    f"Company: {row['company_name']}\nDate: {row['Date']}\nArticle: {row['Document']}"
    for _, row in df.iterrows()
]

metadatas = df.to_dict(orient="records")
ids = [str(i) for i in range(len(df))]

collection.upsert(documents=documents, metadatas=metadatas, ids=ids)

print("DB successfully built!")