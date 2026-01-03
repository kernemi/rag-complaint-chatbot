import pandas as pd
import numpy as np
import chromadb
from chromadb.config import Settings
import faiss

def load_parquet_embeddings(parquet_path: str):
    """
    Load pre-built complaint embeddings from parquet.
    Returns:
        - embeddings: list of numpy arrays
        - texts: list of text chunks
        - metadatas: list of dicts
    """
    df = pd.read_parquet(parquet_path)
    
    # Convert embedding column to list of floats
    embeddings = np.vstack(df['embedding'].to_numpy())
    
    texts = df['document'].tolist()
    
    # Extract metadata for each chunk
    metadatas = df.drop(columns=['embedding', 'document']).to_dict(orient='records')
    
    return embeddings, texts, metadatas

def build_chroma_from_parquet(
    embeddings, 
    texts, 
    metadatas, 
    persist_dir="../vector_store", 
    collection_name="complaints_full",
    batch_size=1000
):
    import chromadb

    # ✅ THIS IS THE KEY FIX
    client = chromadb.PersistentClient(path=persist_dir)

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    ids = [f"chunk_{i}" for i in range(len(texts))]

    for i in range(0, len(texts), batch_size):
        collection.add(
            documents=texts[i:i + batch_size],
            embeddings=embeddings[i:i + batch_size].tolist(),
            metadatas=metadatas[i:i + batch_size],
            ids=ids[i:i + batch_size]
        )

        print(f"Added {i} → {min(i + batch_size, len(texts))}")

        if i >= 10000:  # testing safety
            break

    print("✅ ChromaDB collection persisted to disk")
    return collection


# optional : Build FAISS Index (if you prefer FAISS over Chroma)
def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index