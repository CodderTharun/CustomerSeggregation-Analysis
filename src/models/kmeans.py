from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

def evaluate_k_range(X, k_range, seed=42):
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
        rows.append({"k": k, "inertia": km.inertia_,
                     "silhouette": silhouette_score(X, km.labels_),
                     "davies_bouldin": davies_bouldin_score(X, km.labels_)})
    return rows

def fit_kmeans(X, k, seed=42):
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)