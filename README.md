# 📝 TASK 1 REPORT

## EDA Summary
The CFPB dataset contains complaints across multiple financial products, with Credit Cards accounting for the largest share. A significant number of complaints lacked narrative text and were removed, as textual content is essential for semantic retrieval. Narrative length analysis revealed high variance, ranging from very short complaints to long multi-paragraph descriptions, motivating a chunking strategy in later stages.

## Preprocessing
The dataset was filtered to include only Credit Cards, Personal Loans, Savings Accounts, and Money Transfers. All complaints without narratives were removed. Text normalization techniques such as lowercasing, removal of boilerplate phrases, and special character stripping were applied to improve embedding quality. The cleaned dataset was saved for downstream embedding and retrieval tasks.