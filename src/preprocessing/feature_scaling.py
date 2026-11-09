import numpy as np, joblib
from sklearn.preprocessing import StandardScaler

def scale_rfm(rfm, cols=("Recency", "Frequency", "Monetary"), save_path=None):
    X = np.log1p(rfm[list(cols)])
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    if save_path:
        joblib.dump(scaler, save_path)
    return Xs, scaler