# 📝 TASK 1 REPORT

## EDA Summary
The CFPB dataset contains complaints across multiple financial products, with Credit Cards accounting for the largest share. A significant number of complaints lacked narrative text and were removed, as textual content is essential for semantic retrieval. Narrative length analysis revealed high variance, ranging from very short complaints to long multi-paragraph descriptions, motivating a chunking strategy in later stages.

## Preprocessing
The dataset was filtered to include only Credit Cards, Personal Loans, Savings Accounts, and Money Transfers. All complaints without narratives were removed. Text normalization techniques such as lowercasing, removal of boilerplate phrases, and special character stripping were applied to improve embedding quality. The cleaned dataset was saved for downstream embedding and retrieval tasks.

# 📝 TASK 2 REPORT

## Sampling Strategy
A stratified sampling approach was used to select 12,000 complaints while preserving the proportional distribution across product categories. This ensured that smaller product groups were not underrepresented during embedding.

## Chunking Strategy
Complaint narratives were chunked using a recursive character-based strategy with a chunk size of 500 characters and an overlap of 50 characters. This balanced semantic completeness with retrieval efficiency.

## Embedding Model Choice
The all-MiniLM-L6-v2 sentence transformer was selected due to its strong semantic performance, low computational cost, and compatibility with the pre-built embeddings used in later stages.

## Vector Store Construction
Using the pre-built embeddings in complaint_embeddings.parquet, we constructed a persistent ChromaDB collection named complaints_full. Each chunk contains the embedding, text, and metadata (product, issue, company, state, etc.). This allows fast semantic search without recomputing embeddings, ensuring reproducibility and efficiency.

## Advantages of Using Pre-Built Embeddings

1. Saves hours of computation for 464K+ complaints
2. Guarantees embedding consistency with downstream RAG pipeline
3. Reduces hardware dependency → everyone in the cohort can run the notebook

# 📝 TASK 3 REPORT  
## Building the RAG Core Logic and Evaluation

### Objective
The objective of Task 3 was to design and evaluate a Retrieval-Augmented Generation (RAG) pipeline capable of answering analytical questions about customer complaints using a full-scale, pre-built vector store.


### Vector Store Loading
Instead of recomputing embeddings, the persisted ChromaDB vector store created in Task 2 was loaded directly from disk. This vector store contains embeddings, text chunks, and structured metadata for the complete filtered complaint dataset. Loading the pre-built index ensured consistency, scalability, and reproducibility across all RAG experiments.


### Retriever Design
The retriever component embeds user questions using the same **all-MiniLM-L6-v2** model employed during vector construction. Cosine similarity search is performed against the ChromaDB collection to retrieve the top-k most relevant complaint chunks (k = 5). This guarantees semantic alignment between queries and stored vectors.


### Prompt Engineering Strategy
A carefully designed prompt template was used to constrain the language model and reduce hallucination. The prompt explicitly instructs the model to:
- Act as a financial analyst assistant for CrediTrust  
- Use only the retrieved complaint excerpts  
- Clearly state when the provided context is insufficient  

This prompt structure significantly improves faithfulness and grounding of generated answers.


### Generator Implementation
The generator component combines the user question and retrieved context into a single prompt, which is then passed to a large language model (e.g., Mistral-7B-Instruct). The model produces a natural-language answer grounded strictly in the retrieved complaint text.


### End-to-End RAG Pipeline
The complete RAG workflow follows these steps:
1. Embed user query  
2. Retrieve top-k relevant complaint chunks  
3. Construct a constrained prompt using retrieved context  
4. Generate an answer using an LLM  

This modular design allows the retrieval, prompting, and generation components to be independently improved or replaced.


### Qualitative Evaluation
A qualitative evaluation was conducted using a curated set of representative questions covering credit cards, mortgage servicing, debt collection, BNPL services, and credit reporting. Each response was manually assessed using a 1–5 quality scale based on:
- Relevance of retrieved context  
- Accuracy of the generated answer  
- Faithfulness to source complaints  

### Evaluation Findings
**Strengths**
- High retrieval relevance due to pre-built embeddings  
- Answers were consistently grounded in real complaint narratives  
- Prompt design effectively reduced hallucination  

**Limitations**
- Broad questions occasionally produced generic responses  
- No reranking or citation weighting mechanism was applied  

**Future Improvements**
- Cross-encoder reranking for improved precision  
- Structured outputs with citations  
- Automated evaluation metrics (e.g., faithfulness scoring)


# 📝 TASK 4 REPORT  
## Creating an Interactive Chat Interface

### Objective
The goal of Task 4 was to build an intuitive, user-friendly interface that allows non-technical users to interact with the RAG system and transparently verify the sources used in each response.


### Interface Framework Selection
Gradio was selected as the interface framework due to its lightweight nature, rapid prototyping capabilities, and strong support for ML-driven applications. It integrates seamlessly with Python-based RAG pipelines.


### Core Functionality
The interface provides:
- A text input box for user questions  
- An **Ask** button to trigger the RAG pipeline  
- A display area for the AI-generated answer  
- A **Clear** button to reset the interaction  

All components are designed for simplicity and accessibility.


### Source Transparency and Trust
To enhance user trust, the application displays the top retrieved complaint text chunks below each generated answer. This allows users to:
- Verify factual grounding  
- Understand how conclusions were formed  
- Build confidence in AI-assisted analysis  

Source transparency is a critical requirement for real-world analytical systems in financial domains.


### System Integration
The Gradio app directly integrates with:
- The persisted ChromaDB vector store  
- The MiniLM embedding model  
