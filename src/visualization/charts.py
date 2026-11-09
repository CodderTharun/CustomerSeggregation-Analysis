import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from config import CHART_TEMPLATE, CHART_HEIGHT


# ==========================================
# 1. CUSTOMER SEGMENT DISTRIBUTION
# ==========================================

def segment_distribution(seg):

    counts = seg["Segment"].value_counts().reset_index()

    counts.columns = ["Segment", "Customers"]

    fig = px.pie(
        counts,
        names="Segment",
        values="Customers",
        title="Customer Distribution by Segment",
        hole=0.4,
        template=CHART_TEMPLATE
    )

    fig.update_layout(height=CHART_HEIGHT)

    return fig


# ==========================================
# 2. RECENCY VS MONETARY
# ==========================================

def recency_monetary_chart(seg):

    fig = px.scatter(
        seg,
        x="Recency",
        y="Monetary",
        color="Segment",
        size="Frequency",
        hover_data=["Customer ID"],
        title="Recency vs Monetary Value",
        template=CHART_TEMPLATE
    )

    fig.update_layout(height=CHART_HEIGHT)

    return fig


# ==========================================
# 3. CUSTOMER FREQUENCY DISTRIBUTION
# ==========================================

def frequency_distribution(seg):

    fig = px.histogram(
        seg,
        x="Frequency",
        color="Segment",
        nbins=30,
        title="Customer Purchase Frequency",
        template=CHART_TEMPLATE
    )

    fig.update_layout(height=CHART_HEIGHT)

    return fig


# ==========================================
# 4. MONETARY VALUE BY SEGMENT
# ==========================================

def monetary_by_segment(seg):

    summary = (
        seg.groupby("Segment")["Monetary"]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        summary,
        x="Segment",
        y="Monetary",
        color="Segment",
        title="Average Customer Spending by Segment",
        template=CHART_TEMPLATE
    )

    fig.update_layout(height=CHART_HEIGHT)

    return fig


# ==========================================
# 5. RECENCY DISTRIBUTION
# ==========================================

def recency_distribution(seg):

    fig = px.box(
        seg,
        x="Segment",
        y="Recency",
        color="Segment",
        title="Recency Distribution by Segment",
        template=CHART_TEMPLATE
    )

    fig.update_layout(height=CHART_HEIGHT)

    return fig


# ==========================================
# 6. RFM CORRELATION HEATMAP
# ==========================================

def rfm_correlation(seg):

    features = ["Recency", "Frequency", "Monetary"]

    correlation = seg[features].corr()

    fig = px.imshow(
        correlation,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        title="RFM Feature Correlation",
        template=CHART_TEMPLATE
    )

    fig.update_layout(height=CHART_HEIGHT)

    return fig


# ==========================================
# 7. CUSTOMER COUNT BY SEGMENT
# ==========================================

def customer_count_chart(seg):

    summary = (
        seg["Segment"]
        .value_counts()
        .reset_index()
    )

    summary.columns = ["Segment", "Customers"]

    fig = px.bar(
        summary,
        x="Segment",
        y="Customers",
        color="Segment",
        title="Number of Customers in Each Segment",
        template=CHART_TEMPLATE
    )

    fig.update_layout(height=CHART_HEIGHT)

    return fig


# ==========================================
# 8. REVENUE CONTRIBUTION BY SEGMENT
# ==========================================

def revenue_contribution(seg):

    summary = (
        seg.groupby("Segment")["Monetary"]
        .sum()
        .reset_index()
    )

    fig = px.pie(
        summary,
        names="Segment",
        values="Monetary",
        title="Revenue Contribution by Segment",
        hole=0.4,
        template=CHART_TEMPLATE
    )

    fig.update_layout(height=CHART_HEIGHT)

    return fig