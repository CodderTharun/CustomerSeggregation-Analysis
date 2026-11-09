import pandas as pd

def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates().dropna(subset=["Customer ID"]).copy()
    df["Invoice"] = df["Invoice"].astype(str)
    df = df[~df["Invoice"].str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Customer ID"] = df["Customer ID"].astype(int).astype(str)
    df["Amount"] = df["Quantity"] * df["Price"]
    return df