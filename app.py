import streamlit as st
import pandas as pd
import plotly.express as px
from src.data.data_cleaner import clean_transactions
from src.analysis.rfm_analysis import compute_rfm
from src.preprocessing.feature_scaling import scale_rfm
from src.models.kmeans import (
    evaluate_k_range,
    fit_kmeans
)
from src.segmentation.segment_classifier import label_segments
from src.business_insights.recommendations import get_recommendations
from src.visualization.charts import (
    segment_distribution,
    recency_monetary_chart,
    frequency_distribution,
    monetary_by_segment,
    recency_distribution,
    rfm_correlation,
    customer_count_chart,
    revenue_contribution
)
from config import (
    APP_TITLE,
    APP_LAYOUT,
    DEFAULT_CLUSTERS,
    MIN_CLUSTERS,
    MAX_CLUSTERS
)
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📊",
    layout=APP_LAYOUT
)

st.title("📊 Customer Segmentation System")

st.markdown("""
Analyze customer purchasing behaviour using
**RFM Analysis and Machine Learning**.

Upload your transaction dataset to identify
valuable customer segments and generate
business recommendations.
""")

st.divider()
st.sidebar.title("⚙️ Configuration")

uploaded_file = st.sidebar.file_uploader(
    "Upload Transaction Dataset",
    type=["csv"]
)
st.sidebar.markdown("---")
st.sidebar.info(
    """
    Supported Dataset:

    UCI Online Retail II

    Required Columns:
    - Invoice
    - Customer ID
    - InvoiceDate
    - Quantity
    - Price
    """
)
if uploaded_file is None:

    st.info(
        "Please upload your transaction-level CSV dataset to begin."
    )

    st.stop()
@st.cache_data(show_spinner=False)
def run_pipeline(file_bytes):

    import io

    df = pd.read_csv(io.BytesIO(file_bytes))

    cleaned_df = clean_transactions(df)

    if cleaned_df.empty:
        raise ValueError(
            "No valid transactions found after cleaning."
        )

    rfm = compute_rfm(cleaned_df)

    if rfm.empty:
        raise ValueError(
            "RFM calculation returned an empty dataset."
        )

    X, scaler = scale_rfm(rfm)

    return cleaned_df, rfm, X, scaler


try:

    with st.spinner("Processing customer transactions..."):

        cleaned_df, rfm, X, scaler = run_pipeline(
            uploaded_file.getvalue()
        )

    st.success("Dataset processed successfully!")

except Exception as e:

    st.error(f"Processing Error: {e}")

    st.stop()
st.header("📁 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Transactions",
    f"{len(cleaned_df):,}"
)

col2.metric(
    "Total Customers",
    f"{len(rfm):,}"
)

col3.metric(
    "Total Features",
    len(rfm.columns)
)

col4.metric(
    "Total Revenue",
    f"{rfm['Monetary'].sum():,.2f}"
)

with st.expander("View Dataset Preview"):

    st.dataframe(
        cleaned_df.head(20),
        use_container_width=True
    )

with st.expander("View RFM Dataset"):

    st.dataframe(
        rfm.head(20),
        use_container_width=True
    )
st.divider()
st.sidebar.header("🤖 Machine Learning")
k = st.sidebar.slider(
    "Number of Customer Segments",
    min_value=MIN_CLUSTERS,
    max_value=MAX_CLUSTERS,
    value=DEFAULT_CLUSTERS
)
st.header("🤖 Model Evaluation")
tab1, tab2, tab3 = st.tabs([
    "Elbow Method",
    "Silhouette Score",
    "Model Information"
])
try:
    evaluation_results = pd.DataFrame(
        evaluate_k_range(
            X,
            range(MIN_CLUSTERS, MAX_CLUSTERS + 1)
        )
    )
except Exception as e:

    st.error(f"Model evaluation failed: {e}")

    st.stop()

with tab1:
    st.subheader("Elbow Method")

    fig = px.line(
        evaluation_results,
        x="k",
        y="inertia",
        markers=True,
        title="Inertia vs Number of Clusters"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.caption(
        "Look for the point where the reduction in inertia begins to slow down."
    )
with tab2:
    st.subheader("Silhouette Score")

    fig = px.line(
        evaluation_results,
        x="k",
        y="silhouette",
        markers=True,
        title="Silhouette Score vs Number of Clusters"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.caption(
        "Higher silhouette scores generally indicate better-separated clusters."
    )

with tab3:
    st.subheader("Selected Model Configuration")

    st.write(f"Algorithm: K-Means")

    st.write(f"Number of Clusters: {k}")

    st.write(f"Number of Customers: {len(rfm)}")

    selected_result = evaluation_results[
        evaluation_results["k"] == k
    ]

    if not selected_result.empty:

        st.metric(
            "Silhouette Score",
            round(
                float(selected_result.iloc[0]["silhouette"]),
                4
            )
        )

        st.metric(
            "Inertia",
            round(
                float(selected_result.iloc[0]["inertia"]),
                2
            )
        )
st.divider()
try:

    model = fit_kmeans(X, k)

    seg, profile = label_segments(
        rfm,
        model.labels_
    )

except Exception as e:

    st.error(f"Clustering failed: {e}")

    st.stop()

st.header("👥 Customer Segmentation")

tab4, tab5, tab6 = st.tabs([
    "Segment Overview",
    "Customer Analysis",
    "Segment Profiles"
])
with tab4:

    col1, col2 = st.columns(2)

    with col1:

        st.plotly_chart(
            segment_distribution(seg),
            use_container_width=True
        )

    with col2:

        st.plotly_chart(
            customer_count_chart(seg),
            use_container_width=True
        )

    st.subheader("Revenue Contribution")

    st.plotly_chart(
        revenue_contribution(seg),
        use_container_width=True
    )
with tab5:

    st.subheader("Recency vs Monetary Analysis")

    st.plotly_chart(
        recency_monetary_chart(seg),
        use_container_width=True
    )

    col1, col2 = st.columns(2)

    with col1:

        st.plotly_chart(
            frequency_distribution(seg),
            use_container_width=True
        )

    with col2:

        st.plotly_chart(
            recency_distribution(seg),
            use_container_width=True
        )

    st.subheader("RFM Correlation")

    st.plotly_chart(
        rfm_correlation(seg),
        use_container_width=True
    )
with tab6:

    st.subheader("Customer Segment Characteristics")

    st.dataframe(
        profile,
        use_container_width=True
    )
st.divider()
st.header("🔎 Customer Search")
if "Customer ID" in seg.columns:

    customer_ids = seg["Customer ID"].astype(str)

    search_id = st.text_input(
        "Enter Customer ID"
    )

    if search_id:

        customer_result = seg[
            customer_ids == search_id.strip()
        ]

        if not customer_result.empty:

            st.success("Customer found!")

            st.dataframe(
                customer_result,
                use_container_width=True
            )

        else:

            st.warning("Customer ID not found.")
st.divider()
st.header("💡 Business Insights & Recommendations")
for segment in seg["Segment"].unique():

    segment_data = seg[
        seg["Segment"] == segment
    ]

    with st.expander(
        f"📌 {segment} ({len(segment_data)} customers)",
        expanded=True
    ):

        st.write(
            f"Customer Count: {len(segment_data)}"
        )

        st.write(
            f"Average Recency: {segment_data['Recency'].mean():.2f} days"
        )

        st.write(
            f"Average Frequency: {segment_data['Frequency'].mean():.2f}"
        )

        st.write(
            f"Average Monetary: {segment_data['Monetary'].mean():.2f}"
        )

        st.markdown("**Recommended Business Strategies**")

        st.write(
            get_recommendations(segment)
        )

st.divider()
st.header("📥 Export Results")
col1, col2 = st.columns(2)
with col1:

    st.download_button(
        label="Download Segmented Customer Data",
        data=seg.to_csv(index=False),
        file_name="customer_segments.csv",
        mime="text/csv"
    )
with col2:

    st.download_button(
        label="Download RFM Data",
        data=rfm.to_csv(index=False),
        file_name="customer_rfm.csv",
        mime="text/csv"
    )
st.divider()
st.markdown(
    """
    <div style="text-align:center">
        <p>Customer Segmentation System</p>
        <p>Powered by Python, Scikit-learn and Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)