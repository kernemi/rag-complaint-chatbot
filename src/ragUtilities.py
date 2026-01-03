import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import textwrap

# -----------------------------
# Load Embedding Model
# -----------------------------
def load_embedding_model():
    """
    Load the same embedding model used in Task 2.
    """
    return SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------
# Load Persisted Vector Store
# -----------------------------
def load_chroma_collection(
    persist_dir="../vector_store",
    collection_name="complaints_full"
):
    """
    Load the persisted ChromaDB collection from disk.
    """
    client = chromadb.PersistentClient(path=persist_dir)

    collection = client.get_collection(name=collection_name)
    return collection


# -----------------------------
# Retriever
# -----------------------------
def retrieve_context(
    question: str,
    collection,
    embedder,
    top_k: int = 5
):
    """
    Embed the user question and retrieve top-k similar chunks.
    """
    query_embedding = embedder.encode(question).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    return documents, metadatas


# -----------------------------
# Prompt Engineering
# -----------------------------
def build_prompt(context_chunks, question):
    """
    Build a robust prompt for the LLM.
    """
    context = "\n\n".join(
        [f"- {textwrap.shorten(c, width=500)}" for c in context_chunks]
    )

    prompt = f"""
You are a financial analyst assistant for CrediTrust.
Your task is to answer questions about customer complaints.
Use ONLY the provided complaint excerpts.
If the context does not contain enough information, say so clearly.

Context:
{context}

Question:
{question}

Answer:
"""
    return prompt.strip()


# -----------------------------
# Generator (LLM)
# -----------------------------
def generate_answer(prompt, llm_pipeline):
    """
    Generate an answer using an LLM pipeline.
    """
    response = llm_pipeline(
        prompt,
        max_new_tokens=300,
        do_sample=False
    )
    return response[0]["generated_text"].split("Answer:")[-1].strip()


# -----------------------------
# End-to-End RAG Pipeline
# -----------------------------
def rag_pipeline(question, collection, embedder, llm_pipeline, top_k=5):
    """
    Full RAG flow: retrieve → prompt → generate.
    """
    docs, metas = retrieve_context(
        question,
        collection,
        embedder,
        top_k=top_k
    )

    prompt = build_prompt(docs, question)
    answer = generate_answer(prompt, llm_pipeline)

    return answer, docs, metas
