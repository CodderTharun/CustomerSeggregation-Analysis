from pathlib import Path

# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

PROJECT_NAME = "Customer Segmentation System"
VERSION = "1.0.0"

BASE_DIR = Path(__file__).resolve().parent


# ==========================================
# 2. DIRECTORY CONFIGURATION
# ==========================================

DATA_DIR = BASE_DIR / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"
FIGURE_DIR = REPORT_DIR / "figures"


# ==========================================
# 3. DATASET CONFIGURATION
# ==========================================

DATASET_NAME = "online_retail_II.csv"

RAW_DATA_PATH = RAW_DATA_DIR / DATASET_NAME

PROCESSED_DATA_PATH = (
    PROCESSED_DATA_DIR / "cleaned_customers.csv"
)

SUPPORTED_FORMATS = [".csv"]


# ==========================================
# 4. DATASET COLUMN CONFIGURATION
# ==========================================

CUSTOMER_ID = "Customer ID"

INVOICE_ID = "Invoice"

INVOICE_DATE = "InvoiceDate"

QUANTITY = "Quantity"

PRICE = "Price"

PRODUCT_CODE = "StockCode"

COUNTRY = "Country"


# ==========================================
# 5. DATA PREPROCESSING CONFIGURATION
# ==========================================

REMOVE_MISSING_CUSTOMERS = True

REMOVE_DUPLICATES = True

REMOVE_CANCELLED_INVOICES = True

REMOVE_NON_POSITIVE_QUANTITIES = True

REMOVE_NON_POSITIVE_PRICES = True


# ==========================================
# 6. RFM CONFIGURATION
# ==========================================

RFM_FEATURES = [
    "Recency",
    "Frequency",
    "Monetary"
]

REFERENCE_DATE = None


# ==========================================
# 7. MACHINE LEARNING CONFIGURATION
# ==========================================

DEFAULT_ALGORITHM = "KMeans"

DEFAULT_CLUSTERS = 4

MIN_CLUSTERS = 2

MAX_CLUSTERS = 8

RANDOM_STATE = 42

N_INIT = 10

MAX_ITER = 300


# ==========================================
# 8. MODEL STORAGE
# ==========================================

KMEANS_MODEL_PATH = MODEL_DIR / "kmeans_model.pkl"

SCALER_PATH = MODEL_DIR / "scaler.pkl"


# ==========================================
# 9. APPLICATION CONFIGURATION
# ==========================================

APP_TITLE = "Customer Segmentation System"

APP_LAYOUT = "wide"

MAX_UPLOAD_SIZE_MB = 200


# ==========================================
# 10. VISUALIZATION CONFIGURATION
# ==========================================

CHART_TEMPLATE = "plotly_white"

CHART_HEIGHT = 500