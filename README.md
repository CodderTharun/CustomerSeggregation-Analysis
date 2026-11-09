# CUSTOMER SEGMENTATION SYSTEM
## Project Technical Documentation

**Technology Stack:** Python, Pandas, NumPy, Scikit-learn, Streamlit, Plotly  
**Machine Learning Approach:** Unsupervised Learning  
**Primary Algorithm:** K-Means Clustering  
**Deployment:** Streamlit Community Cloud

---

# 1. Project Overview

## 1.1 Introduction

Customer Segmentation is a Data Science and Machine Learning project that divides customers into different groups based on their purchasing behaviour, spending patterns, demographics and transaction history.

Businesses often have thousands or millions of customers. Treating every customer equally can result in ineffective marketing campaigns, poor customer engagement and missed business opportunities.

This project aims to develop an intelligent customer segmentation platform that automatically analyses customer information and identifies groups of customers with similar characteristics.

The application will allow users to upload customer datasets, perform data preprocessing, analyse customer behaviour, execute machine learning algorithms and visualize the resulting customer segments through an interactive dashboard.

## 1.2 Problem Statement

Businesses collect large amounts of customer information, but extracting meaningful patterns from this data is challenging.

Some common problems include:

- Difficulty identifying high-value customers.
- Lack of understanding of customer purchasing behaviour.
- Ineffective marketing campaigns.
- Difficulty identifying inactive customers.
- Inability to identify customers with similar purchasing patterns.
- Lack of data-driven marketing strategies.

The proposed system addresses these problems through automated customer segmentation using machine learning.

## 1.3 Project Objectives

1. Develop a platform to upload and analyse customer datasets.
2. Perform automated data cleaning and preprocessing.
3. Identify customer behaviour patterns using Exploratory Data Analysis.
4. Calculate meaningful customer metrics using RFM analysis.
5. Implement clustering algorithms to group similar customers.
6. Evaluate the quality of generated customer segments.
7. Generate meaningful customer categories.
8. Provide visual representations of customer segments.
9. Generate business recommendations based on customer behaviour.
10. Deploy the application as a publicly accessible web application.

---

# 2. System Architecture

The project follows a modular monolithic architecture.

Each component is responsible for a specific operation, making the application easier to maintain, test and extend.

## 2.1 High-Level Architecture

USER
  |
  v
STREAMLIT WEB APPLICATION
  |
  v
DATASET UPLOAD
  |
  v
DATA LOADING MODULE
  |
  v
DATA VALIDATION
  |
  v
DATA CLEANING
  |
  v
EXPLORATORY DATA ANALYSIS
  |
  v
FEATURE ENGINEERING
  |
  v
RFM ANALYSIS
  |
  v
FEATURE SCALING
  |
  v
MACHINE LEARNING ENGINE
  |
  v
CLUSTERING EVALUATION
  |
  v
CUSTOMER SEGMENT IDENTIFICATION
  |
  v
BUSINESS INSIGHTS GENERATION
  |
  v
INTERACTIVE DASHBOARD
  |
  v
DOWNLOAD SEGMENTED DATASET / REPORT

## 2.2 Architecture Components

| Component | Responsibility |
|---|---|
| Frontend | Provides user interaction through Streamlit |
| Data Loader | Reads uploaded customer datasets |
| Data Cleaner | Handles missing values, duplicates and invalid records |
| EDA Engine | Analyses customer behaviour and data distributions |
| Feature Engineering | Creates meaningful customer attributes |
| RFM Engine | Calculates Recency, Frequency and Monetary values |
| ML Engine | Performs customer clustering |
| Evaluation Engine | Evaluates clustering quality |
| Segmentation Engine | Assigns meaningful customer categories |
| Visualization Engine | Creates charts and analytical dashboards |
| Business Insights Engine | Generates business recommendations |

---

# 3. Dataset Understanding

The system should support customer datasets in CSV format.

Two types of datasets can be supported.

## 3.1 Customer-Level Dataset

This dataset contains one record per customer.

Example:

| Customer ID | Age | Gender | Annual Income | Spending Score |
|---|---:|---|---:|---:|
| C001 | 22 | Male | 45000 | 75 |
| C002 | 35 | Female | 65000 | 45 |
| C003 | 28 | Male | 30000 | 85 |

This dataset is suitable for basic demographic and spending-based segmentation.

## 3.2 Transaction-Level Dataset

This dataset contains individual customer transactions.

Example:

| Transaction ID | Customer ID | Date | Amount |
|---|---|---|---:|
| T001 | C001 | 2026-01-10 | 2500 |
| T002 | C002 | 2026-01-12 | 1500 |
| T003 | C001 | 2026-02-05 | 3500 |

This dataset is more useful for behavioural segmentation.

It allows the system to calculate:

- Total customer spending.
- Number of transactions.
- Average transaction value.
- Last purchase date.
- Purchase frequency.

**Important:** RFM analysis requires transaction-level information or equivalent precomputed customer metrics. It cannot be accurately calculated from a dataset containing only age and annual income.

---

# 4. Detailed Module Working

## 4.1 Dataset Upload and Data Loading

File: `src/data/data_loader.py`

The first module handles the dataset uploaded by the user.

### Working Process

1. User opens the Streamlit application.
2. User uploads a CSV file.
3. The application reads the uploaded file.
4. Pandas converts the CSV into a DataFrame.
5. The system checks the dataset structure.
6. The dataset is passed to the validation module.

### Responsibilities

- Read CSV files.
- Identify column names.
- Identify data types.
- Display dataset previews.
- Validate mandatory columns.
- Identify unsupported data formats.

### Output

A Pandas DataFrame containing the uploaded customer information.

---

## 4.2 Data Validation and Cleaning

File: `src/data/data_cleaner.py`

Real-world customer datasets may contain incomplete or inconsistent information.

Before applying machine learning, the system must clean the dataset.

### Operations

**A. Missing Value Detection**

Identify columns containing missing values.

Possible handling strategies:

- Numerical columns: Mean or median imputation.
- Categorical columns: Mode imputation.
- Records with excessive missing information: Remove when justified.

**B. Duplicate Removal**

Identify and remove duplicate customer or transaction records.

**C. Data Type Conversion**

Convert columns into appropriate data types.

Examples:

- Transaction date into datetime.
- Customer ID into string.
- Purchase amount into numerical format.

**D. Outlier Detection**

Identify unusually high or low numerical values.

Possible methods:

- Interquartile Range (IQR).
- Z-Score.
- Statistical distribution analysis.

Outliers should not automatically be deleted. High-value customers may legitimately have unusually large purchases.

### Output

A cleaned and validated customer dataset.

---

## 4.3 Exploratory Data Analysis (EDA)

File: `src/analysis/eda.py`

EDA is performed to understand customer behaviour before building the machine learning model.

### Analysis Performed

1. Customer age distribution.
2. Annual income distribution.
3. Spending distribution.
4. Purchase frequency.
5. Average purchase amount.
6. Customer demographics.
7. Correlation between numerical features.

### Visualizations

- Histograms.
- Box plots.
- Scatter plots.
- Correlation heatmaps.
- Bar charts.

### Purpose

EDA helps identify meaningful features and understand the relationships between customer attributes.

For example:

If customers with higher income consistently demonstrate higher spending, income may be an important feature for segmentation.

### Output

Statistical summaries and visual representations of the dataset.

---

## 4.4 Feature Engineering

File: `src/preprocessing/feature_engineering.py`

Feature engineering converts raw customer information into meaningful numerical attributes that machine learning algorithms can understand.

### Example

Original transaction data:

| Customer | Transaction Amount |
|---|---:|
| C001 | 2000 |
| C001 | 3000 |
| C001 | 1500 |
| C002 | 1000 |
| C002 | 2000 |

After feature engineering:

| Customer | Total Spending | Purchase Count | Average Order Value |
|---|---:|---:|---:|
| C001 | 6500 | 3 | 2166.67 |
| C002 | 3000 | 2 | 1500 |

### Features Generated

- Total Spending.
- Purchase Frequency.
- Average Order Value.
- Days Since Last Purchase.
- Customer Age.
- Annual Income.

These features provide a better representation of customer behaviour.

---

# 5. RFM Analysis

File: `src/analysis/rfm_analysis.py`

RFM analysis is one of the major components of this project.

RFM stands for:

1. Recency.
2. Frequency.
3. Monetary.

It helps understand customer purchasing behaviour.

## 5.1 Recency

Recency represents how recently a customer made a purchase.

Calculation:

Recency = Reference Date - Last Purchase Date

Example:

Customer A purchased 5 days ago.

Customer B purchased 90 days ago.

Customer A has better recency.

## 5.2 Frequency

Frequency represents how many purchases a customer has made during a specified period.

Example:

Customer A: 25 purchases.

Customer B: 3 purchases.

Customer A has higher purchase frequency.

## 5.3 Monetary

Monetary represents the total amount spent by a customer.

Example:

Customer A: ₹50,000.

Customer B: ₹2,000.

Customer A has higher monetary value.

## 5.4 RFM Output

| Customer | Recency | Frequency | Monetary |
|---|---:|---:|---:|
| C001 | 5 | 25 | 50000 |
| C002 | 90 | 3 | 2000 |
| C003 | 10 | 15 | 25000 |

The RFM output becomes an important input for the clustering algorithm.

---

# 6. Feature Scaling

File: `src/preprocessing/feature_scaling.py`

Machine learning algorithms such as K-Means use distance calculations.

Different numerical features may have significantly different ranges.

Example:

Annual Income: 10,000–1,00,000.

Spending Score: 1–100.

Without scaling, large numerical ranges may disproportionately influence distance calculations.

### Solution: StandardScaler

StandardScaler transforms features into a standardized scale.

The formula is:

Z = (X - Mean) / Standard Deviation

Where:

- X = Original feature value.
- Mean = Mean of the feature.
- Standard Deviation = Standard deviation of the feature.
- Z = Standardized value.

### Output

A scaled numerical feature matrix that can be passed to the clustering algorithm.

The scaler should be saved and reused when assigning segments to new customers.

---

# 7. Machine Learning Engine

Directory: `src/models/`

The machine learning engine identifies groups of customers with similar characteristics.

The project primarily uses unsupervised learning.

## 7.1 Why Unsupervised Learning?

In traditional supervised learning, the model learns from predefined labels.

However, customer datasets generally do not contain predefined customer segment labels.

Therefore, unsupervised learning is appropriate because the algorithm discovers patterns directly from the dataset.

## 7.2 K-Means Clustering

File: `src/models/kmeans.py`

K-Means is the primary clustering algorithm.

### Working Process

Step 1: Select the number of clusters (K).

Step 2: Initialize K centroids.

Step 3: Calculate the distance between each customer and the centroids.

Step 4: Assign each customer to the nearest centroid.

Step 5: Recalculate centroid positions.

Step 6: Repeat the process until the clusters stabilize.

### Example

Suppose there are 1,000 customers.

We select K = 5.

The algorithm divides the customers into five groups based on their similarities.

The resulting clusters might look like:

- Cluster 0.
- Cluster 1.
- Cluster 2.
- Cluster 3.
- Cluster 4.

These cluster numbers do not have business meanings yet.

They are simply identifiers generated by the algorithm.

## 7.3 Selecting the Optimal K

The system should support the Elbow Method.

The Elbow Method evaluates different cluster counts and calculates the Within-Cluster Sum of Squares (inertia).

As the number of clusters increases, inertia generally decreases.

The appropriate K can be selected by examining where the improvement starts diminishing.

The Silhouette Score can also be used to evaluate how well-separated the clusters are.

## 7.4 Additional Algorithms

The application can later support:

**Hierarchical Clustering**

Creates a hierarchy of customer groups and can be useful for exploring relationships between clusters.

**DBSCAN**

Identifies dense customer groups and can detect noise or outliers without requiring a predefined number of clusters.

These algorithms can be compared against K-Means.

---

# 8. Model Evaluation

File: `src/models/model_evaluation.py`

After clustering, the system needs to evaluate whether the generated customer groups are meaningful.

### Evaluation Metrics

**1. Silhouette Score**

Measures how similar a customer is to its own cluster compared with other clusters.

Its value ranges from -1 to +1.

Higher values generally indicate better-separated clusters.

**2. Davies-Bouldin Index**

Measures the similarity between different clusters.

Lower values generally indicate better clustering.

**3. Inertia**

Measures the sum of squared distances between data points and their assigned cluster centroids.

Lower inertia alone does not guarantee better clustering.

### Output

The evaluation module provides metrics that help users compare different clustering configurations.

---

# 9. Customer Segment Identification

File: `src/segmentation/segment_classifier.py`

K-Means produces numerical cluster identifiers.

The segmentation module converts these identifiers into meaningful business categories.

### Example

Suppose the clustering algorithm produces the following cluster summary:

| Cluster | Average Recency | Average Frequency | Average Monetary |
|---|---:|---:|---:|
| 0 | 5 | 25 | 50000 |
| 1 | 90 | 3 | 2000 |
| 2 | 10 | 15 | 25000 |
| 3 | 60 | 8 | 8000 |

The segmentation module analyses these statistics.

It may generate the following labels:

| Cluster | Customer Segment |
|---|---|
| 0 | Premium Customers |
| 1 | At-Risk Customers |
| 2 | Loyal Customers |
| 3 | Regular Customers |

These labels must be determined using cluster characteristics and business rules.

They should not be assigned randomly or based only on cluster numbers.

### Output

Every customer receives a meaningful segment label.

---

# 10. Business Insights Generation

File: `src/business_insights/recommendations.py`

This module converts machine learning results into actionable business recommendations.

The objective is not just to identify customer groups but also to help businesses make decisions.

### Example Recommendations

**Premium Customers**

Characteristics:
- High spending.
- Frequent purchases.
- Recent transactions.

Recommendation:
- Provide exclusive offers.
- Introduce VIP membership.
- Offer early access to new products.

**At-Risk Customers**

Characteristics:
- Long time since the last purchase.
- Previously active customers.
- Declining purchase activity.

Recommendation:
- Send re-engagement campaigns.
- Provide personalized discounts.
- Introduce loyalty incentives.

**Budget Customers**

Characteristics:
- Low spending.
- Lower average order value.

Recommendation:
- Offer affordable product bundles.
- Provide discounts on selected products.
- Recommend products based on purchase history.

### Implementation Approach

Initially, use a rule-based recommendation engine.

For example:

IF customer segment = Premium:

    Recommend VIP rewards.

IF customer segment = At-Risk:

    Recommend re-engagement campaigns.

IF customer segment = Budget:

    Recommend discount offers.

Later, an LLM can be integrated to generate more detailed natural-language business reports.

---

# 11. Visualization Engine

Directory: `src/visualization/`

The visualization engine presents the analytical results through interactive charts.

Plotly is recommended for interactive visualizations.

### Required Visualizations

1. Customer distribution by segment.
2. Income versus spending.
3. Recency versus frequency.
4. Monetary value by customer segment.
5. Cluster distribution.
6. Customer demographic distribution.
7. Average spending by segment.

### Example

A scatter plot can display:

X-axis: Annual Income.

Y-axis: Spending Score.

Each point represents a customer.

Different colours represent different clusters.

This helps users visually understand how the clustering algorithm divides customers.

---

# 12. Streamlit Dashboard

File: `app.py`

Streamlit acts as the frontend of the application.

Users do not need programming knowledge to interact with the system.

## 12.1 Dashboard Pages

### Page 1: Overview

Displays:

- Total customers.
- Number of customer segments.
- Average customer spending.
- Average customer income.
- Customer distribution.

### Page 2: Dataset Management

Allows users to:

- Upload CSV files.
- Preview datasets.
- View dataset statistics.
- Check missing values.
- Review data quality.

### Page 3: Data Analysis

Displays:

- Customer demographics.
- Purchase patterns.
- Spending distribution.
- Correlation analysis.

### Page 4: Customer Segmentation

Allows users to:

- Select clustering algorithms.
- Configure the number of clusters.
- Execute clustering.
- Evaluate the clustering results.
- View customer segment assignments.

### Page 5: Customer Insights

Displays:

- Customer segment characteristics.
- Segment-wise spending.
- Purchase frequency.
- Business recommendations.

### Page 6: Reports

Allows users to:

- Download segmented customer datasets.
- Export analytical summaries.
- Generate PDF reports.

---

# 13. Complete Application Workflow

The following represents the complete execution flow.

START
  |
  v
User opens the application
  |
  v
User uploads customer dataset
  |
  v
Validate uploaded file
  |
  +---- Invalid ----> Display validation error
  |                         |
  |                         v
  |                    Re-upload dataset
  |
  v
Load dataset
  |
  v
Perform data cleaning
  |
  v
Generate EDA statistics
  |
  v
Perform feature engineering
  |
  v
Calculate RFM metrics
  |
  v
Select relevant features
  |
  v
Apply feature scaling
  |
  v
Select clustering algorithm
  |
  v
Determine cluster configuration
  |
  v
Execute clustering
  |
  v
Evaluate clustering performance
  |
  +---- Unsatisfactory ----> Adjust parameters
  |                                |
  |                                v
  |                         Execute clustering again
  |
  v
Generate customer clusters
  |
  v
Analyse cluster characteristics
  |
  v
Assign customer segment labels
  |
  v
Generate business recommendations
  |
  v
Display interactive dashboard
  |
  v
Export segmented customer dataset
  |
  v
END

---

# 14. Database Integration (Future Enhancement)

The initial version can work without a database.

However, for a more advanced application, PostgreSQL can be integrated.

### Proposed Tables

**1. Users**

Stores application user information.

- User ID.
- Username.
- Email.
- Password Hash.
- Created Date.

**2. Datasets**

Stores uploaded dataset metadata.

- Dataset ID.
- User ID.
- Dataset Name.
- Upload Date.
- Record Count.

**3. Segmentation Results**

Stores customer segmentation results.

- Result ID.
- Dataset ID.
- Customer ID.
- Cluster ID.
- Segment Name.

**4. Model History**

Stores machine learning execution details.

- Model ID.
- Algorithm.
- Number of Clusters.
- Evaluation Score.
- Execution Date.

### Why Integrate a Database?

- Maintain historical segmentation results.
- Track multiple datasets.
- Support multiple users.
- Compare model performance.
- Store customer analysis history.

---

# 15. Deployment Architecture

The initial version will be deployed using Streamlit Community Cloud.

### Deployment Workflow

Local Development
  |
  v
Python Application
  |
  v
Git Repository
  |
  v
GitHub Repository
  |
  v
Streamlit Community Cloud
  |
  v
Dependency Installation
  |
  v
Application Deployment
  |
  v
Public Application URL

### Deployment Requirements

- GitHub repository.
- requirements.txt.
- app.py.
- All required Python modules.
- Streamlit-compatible configuration.

The application can then be accessed through a public URL.

### Future Production Deployment

For a more advanced deployment architecture:

Frontend: React or Streamlit.

Backend: FastAPI.

Database: PostgreSQL.

Machine Learning: Scikit-learn.

Containerization: Docker.

Cloud: AWS / Azure.

This architecture is optional and should be considered after the initial application is working.

---

# 16. Testing Strategy

The project should include unit testing and integration testing.

### Unit Testing

Test individual modules independently.

Examples:

- Data loader correctly reads CSV files.
- Data cleaner handles missing values.
- Feature engineering generates correct metrics.
- RFM calculations produce expected values.
- Segmentation module assigns valid labels.

### Integration Testing

Test the complete pipeline:

Dataset Upload → Cleaning → Feature Engineering → Clustering → Segmentation → Dashboard.

### Performance Testing

Evaluate:

- Dataset processing time.
- Model execution time.
- Dashboard loading time.
- Memory consumption.

---

# 17. Security and Data Privacy

Customer datasets may contain sensitive business and personal information.

The application should follow basic data protection practices.

1. Validate uploaded file types.
2. Avoid executing uploaded files.
3. Restrict access to uploaded customer information.
4. Do not expose customer information through public GitHub repositories.
5. Avoid storing passwords in plain text.
6. Use environment variables for credentials.
7. Delete temporary datasets when no longer required.
8. Use HTTPS for deployed applications.

For production use, authentication, authorization, secure storage and appropriate data retention policies should be implemented.

---

# 18. Expected Project Outcomes

After successful implementation, the system should provide:

1. Automated customer data processing.
2. Customer behaviour analysis.
3. RFM-based customer evaluation.
4. Machine learning-based customer segmentation.
5. Customer segment classification.
6. Clustering performance evaluation.
7. Interactive customer analytics dashboard.
8. Business recommendations.
9. Downloadable segmentation reports.

---

# 19. Future Enhancements

The following features can make the project more advanced.

| Enhancement | Purpose |
|---|---|
| Customer Churn Prediction | Identify customers likely to stop purchasing |
| Customer Lifetime Value | Estimate long-term customer value |
| Recommendation System | Recommend products to customers |
| Automated Model Selection | Compare clustering algorithms |
| LLM Integration | Generate natural-language business reports |
| Scheduled Segmentation | Automatically process new transaction data |
| Database Integration | Maintain historical analysis |
| Authentication | Support multiple business users |
| Model Monitoring | Track changes in customer behaviour |

---

# 20. Conclusion

The Customer Segmentation System is an end-to-end Data Science application that combines data preprocessing, exploratory analysis, feature engineering, RFM analysis, unsupervised machine learning and interactive visualization.

The system transforms raw customer information into meaningful customer groups and provides actionable business recommendations.

The modular architecture allows individual components to be developed, tested and maintained independently.

The initial implementation will use Python, Pandas, Scikit-learn and Streamlit. Future enhancements can introduce FastAPI, PostgreSQL, Docker, cloud deployment and LLM-based business intelligence.

The final objective is to build a practical customer analytics platform that demonstrates the complete Data Science lifecycle, from raw data ingestion to business decision-making.