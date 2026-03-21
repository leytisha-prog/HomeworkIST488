import streamlit as st
import pandas as pd 
import chromadb


# 1. File Upload
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of Data:", df.head())

# 2. Initialize ChromaDB Client 
client = chromadb.PersistentClient(path="./chroma_db_data")
collection = client.get_or_create_collection(name="csv_collection")

if st.button("Add to ChromaDB"):
    # Convert DataFrame to lists for Chroma 
    documents = df['body_column'].astype(str).tolist()
    metadatas = df.drop(columns=['body_column']).to_dict(orient='records')
    ids = [str(i) for i in range(len(df))]

    # Insert Data
    collection.upsert(
        documents=documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    st.success(f"Successfully added {len(df)} rows to ChromaDB!")