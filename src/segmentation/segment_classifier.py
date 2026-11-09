def label_segments(rfm, labels):
    rfm = rfm.copy()
    rfm["Cluster"] = labels
    prof = rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()

    # rank-based scoring: low recency is good, high F/M is good
    score = (prof["Recency"].rank(ascending=False)
             + prof["Frequency"].rank() + prof["Monetary"].rank())
    order = score.sort_values(ascending=False).index.tolist()

    names = ["Premium Customers", "Loyal Customers",
             "Regular Customers", "At-Risk Customers"]
    names += [f"Segment {i}" for i in range(len(order) - len(names))]
    mapping = {c: names[i] for i, c in enumerate(order)}
    rfm["Segment"] = rfm["Cluster"].map(mapping)
    return rfm, prof