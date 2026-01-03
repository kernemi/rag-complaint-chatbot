# 📝 TASK 1 REPORT

## EDA Summary
The CFPB dataset contains complaints across multiple financial products, with Credit Cards accounting for the largest share. A significant number of complaints lacked narrative text and were removed, as textual content is essential for semantic retrieval. Narrative length analysis revealed high variance, ranging from very short complaints to long multi-paragraph descriptions, motivating a chunking strategy in later stages.

## Preprocessing
The dataset was filtered to include only Credit Cards, Personal Loans, Savings Accounts, and Money Transfers. All complaints without narratives were removed. Text normalization techniques such as lowercasing, removal of boilerplate phrases, and special character stripping were applied to improve embedding quality. The cleaned dataset was saved for downstream embedding and retrieval tasks.

# 📝 TASK 2 REPORT SECTION

## Sampling Strategy
A stratified sampling approach was used to select 12,000 complaints while preserving the proportional distribution across product categories. This ensured that smaller product groups were not underrepresented during embedding.

## Chunking Strategy
Complaint narratives were chunked using a recursive character-based strategy with a chunk size of 500 characters and an overlap of 50 characters. This balanced semantic completeness with retrieval efficiency.

## Embedding Model Choice
The all-MiniLM-L6-v2 sentence transformer was selected due to its strong semantic performance, low computational cost, and compatibility with the pre-built embeddings used in later stages.