import pandas as pd

def load_dataset():

    file_path = "data/raw/online_retail.csv"

    df = pd.read_csv(file_path)

    return df