import streamlit as st

st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="📊",
    layout="wide"
)

st.title("Customer Segmentation System")

st.markdown("""
### Welcome!

An intelligent customer segmentation platform
powered by Machine Learning.

Upload your customer dataset to begin.
""")

uploaded_file = st.file_uploader(
    "Upload Customer Dataset",
    type=["csv"]
)

if uploaded_file:

    import pandas as pd

    df = pd.read_csv(uploaded_file)

    st.success("Dataset uploaded successfully!")

    st.subheader("Dataset Preview")

    st.dataframe(df.head())

    st.subheader("Dataset Information")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Customers", len(df))

    col2.metric("Total Columns", len(df.columns))

    col3.metric("Missing Values", int(df.isnull().sum().sum()))