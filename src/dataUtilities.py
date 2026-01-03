import pandas as pd
import re


def load_data(path: str) -> pd.DataFrame:
    """
    Load CFPB complaint dataset from CSV file.
    """
    return pd.read_csv(path)


def filter_products_and_narratives(
    df: pd.DataFrame,
    valid_products: list
) -> pd.DataFrame:
    """
    Filter dataset to include only selected products
    and non-empty complaint narratives.
    """
    filtered_df = df[
        (df["Product"].isin(valid_products)) &
        (df["Consumer complaint narrative"].notna())
    ].copy()

    return filtered_df


def compute_narrative_length(
    df: pd.DataFrame,
    text_column: str
) -> pd.DataFrame:
    """
    Add a word-count column for complaint narratives.
    """
    df["narrative_length"] = (
        df[text_column]
        .astype(str)
        .apply(lambda x: len(x.split()))
    )
    return df


def clean_text(text: str) -> str:
    """
    Clean complaint narrative text for NLP processing.
    """
    text = text.lower()
    text = re.sub(r"i am writing to file a complaint", "", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def apply_text_cleaning(
    df: pd.DataFrame,
    text_column: str,
    new_column: str = "cleaned_narrative"
) -> pd.DataFrame:
    """
    Apply text cleaning to narrative column.
    """
    df[new_column] = df[text_column].apply(clean_text)
    return df


def save_dataframe(
    df: pd.DataFrame,
    output_path: str
) -> None:
    """
    Save DataFrame to CSV.
    """
    df.to_csv(output_path, index=False)
