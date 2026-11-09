import pandas as pd

def compute_rfm(df, cust="Customer ID", date="InvoiceDate",
                inv="Invoice", amt="Amount"):
    ref = df[date].max() + pd.Timedelta(days=1)
    rfm = df.groupby(cust).agg(
        Recency=(date, lambda x: (ref - x.max()).days),
        Frequency=(inv, "nunique"),
        Monetary=(amt, "sum"),
    ).reset_index()
    return rfm