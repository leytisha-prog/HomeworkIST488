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