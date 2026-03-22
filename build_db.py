import sys
__import__("pysqlite3")
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import os
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

BASE_DIR = os.path.dirname(__file__)

df = pd.read_csv(
    os.path.join(BASE_DIR, "HWs", "news.csv"),
    sep=",",   
    engine="python",
    on_bad_lines="skip"
)
df.columns = df.columns.str.strip()

print("Columns:", df.columns)

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.environ["OPENAI_API_KEY"],
    model_name="text-embedding-3-small"
)

client = chromadb.PersistentClient(
    path=os.path.join(BASE_DIR, "chroma_db_data")
)

collection = client.get_or_create_collection(
    name="news_collection",
    embedding_function=openai_ef
)

documents = [
    f"Company: {row['company_name']}\nDate: {row['Date']}\nArticle: {row['Document']}"
    for _, row in df.iterrows()
]

metadatas = [
    {
        "company": str(row["company_name"]),
        "date": str(row["Date"]),
        "url": str(row["URL"])
    }
    for _, row in df.iterrows()
]

ids = [str(i) for i in range(len(df))]

collection.upsert(
    documents=documents,
    metadatas=metadatas,
    ids=ids
)

print("Rows loaded:", len(df))
print("Collection count after upsert:", collection.count())
print("Collections:", client.list_collections())
print("DB successfully built!")

