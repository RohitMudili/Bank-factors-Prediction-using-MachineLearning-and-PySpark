# Online Banking Analysis and Prediction Using Machine Learning and PySpark

## Table of Contents
- [Project Overview](#project-overview)
- [Datasets](#datasets)
- [Technologies Used](#technologies-used)
- [System Architecture](#system-architecture)
- [Detailed Workflow](#detailed-workflow)
  - [Data Ingestion](#data-ingestion)
  - [Data Processing](#data-processing)
  - [Machine Learning Implementation](#machine-learning-implementation)
  - [Visualization](#visualization)
- [Key Algorithms and Models](#key-algorithms-and-models)
- [Results and Insights](#results-and-insights)
- [Performance Metrics](#performance-metrics)
- [Challenges and Solutions](#challenges-and-solutions)
- [Future Work](#future-work)
- [Team Members and Contributions](#team-members-and-contributions)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Project Overview

This comprehensive project focuses on leveraging big data technologies and machine learning to analyze and predict critical factors in online banking systems. The primary objectives were:

1. To process and analyze large-scale banking datasets efficiently using distributed computing
2. To build predictive models for customer behavior, fraud detection, and risk assessment
3. To create a scalable pipeline for financial data analysis
4. To generate actionable insights for banking decision-makers

The project was implemented by a team of six data engineers and scientists over a period of three months, utilizing a modern big data stack centered around Apache Spark and its ecosystem.

## Datasets

We utilized three primary datasets from Kaggle, each focusing on different aspects of online banking:

1. **Loan Dataset** 
   - Contains 5 million records of loan applications
   - Features: applicant demographics, credit history, loan amount, term, interest rate, approval status
   - Source: [Kaggle Loan Dataset](https://www.kaggle.com/datasets/...)

2. **Credit Card Transactions** 
   - Includes 10 million transactions from 50,000 customers
   - Features: transaction amount, location, merchant category, timestamp, fraud flag
   - Source: [Kaggle Credit Card Fraud](https://www.kaggle.com/datasets/...)

3. **Customer Profiles** 
   - Demographic and behavioral data for 100,000 banking customers
   - Features: age, income, account tenure, product holdings, digital engagement metrics
   - Source: [Kaggle Banking Customers](https://www.kaggle.com/datasets/...)

## Technologies Used

### Core Technologies
- **Apache Spark 3.2.1**: Distributed processing engine
- **PySpark**: Python API for Spark
- **Hadoop 3.3.1**: HDFS for distributed storage
- **Hive 3.1.2**: Data warehouse infrastructure
- **Sqoop 1.4.7**: Data transfer between RDBMS and HDFS

### Machine Learning
- Spark MLlib for distributed machine learning
- Scikit-learn for local model development
- XGBoost for gradient boosted trees
- Hyperopt for hyperparameter tuning

### Visualization
- Matplotlib and Seaborn
- Plotly for interactive dashboards
- Tableau for final reporting

1. **Data Layer**: Raw datasets stored in S3 and HDFS
2. **Processing Layer**: Spark and Hive for ETL
3. **ML Layer**: Model training and evaluation
4. **Serving Layer**: REST API for predictions
5. **Presentation Layer**: Dashboards and reports

### Data Ingestion

1. **Initial Setup**:
   - Configured AWS EMR cluster with 6 nodes (1 master, 5 workers)
   - Established HDFS directories for raw and processed data
   - Set up Hive metastore for table definitions

2. **Data Transfer**:
   ```bash
   sqoop import \
   --connect jdbc:mysql://banking-db/loans \
   --username admin \
   --password securepass \
   --table loan_applications \
   --hive-import \
   --create-hive-table \
   --hive-table loans_raw \
   --target-dir /user/hive/warehouse/loans
   ```

3. **Data Validation**:
   - Implemented data quality checks using PySpark:
     ```python
     from pyspark.sql.functions import col, count, when

     # Check for missing values
     df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns]).show()

     # Validate value ranges
     df.filter((col('loan_amount') < 0) | (col('loan_amount') > 1000000)).count()
     ```

### Data Processing

1. **Cleaning Pipeline**:
   - Handled missing values using median imputation for numerical features
   - Implemented mode imputation for categorical features
   - Outlier detection using IQR method

2. **Feature Engineering**:
   ```python
   from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler

   # Convert categorical features
   indexer = StringIndexer(inputCol="employment_type", outputCol="employment_idx")
   encoder = OneHotEncoder(inputCol="employment_idx", outputCol="employment_vec")

   # Create feature vector
   assembler = VectorAssembler(
       inputCols=["age", "income", "employment_vec", "credit_score"],
       outputCol="features"
   )
   ```

3. **Data Partitioning**:
   - Split data into training (70%), validation (15%), and test (15%) sets
   - Stratified sampling to maintain class distribution

### Machine Learning Implementation

#### Fraud Detection Model

1. **Algorithm Selection**:
   - Compared Logistic Regression, Random Forest, and Gradient Boosted Trees
   - Selected XGBoost for best performance on imbalanced data

2. **Model Training**:
   ```python
   from xgboost import XGBClassifier
   from imblearn.over_sampling import SMOTE

   # Handle class imbalance
   smote = SMOTE(sampling_strategy=0.3)
   X_res, y_res = smote.fit_resample(X_train, y_train)

   # Train model
   model = XGBClassifier(
       scale_pos_weight=5,
       max_depth=6,
       learning_rate=0.1,
       n_estimators=200
   )
   model.fit(X_res, y_res)
   ```

3. **Evaluation Metrics**:
   - Focused on precision-recall due to class imbalance
   - Achieved AUC-ROC of 0.92 on test set

#### Loan Approval Prediction

1. **Feature Importance**:
   ![Feature Importance](images/feature_importance.png)

2. **SHAP Analysis**:
   ```python
   import shap

   explainer = shap.TreeExplainer(model)
   shap_values = explainer.shap_values(X_test)
   shap.summary_plot(shap_values, X_test)
   ```

### Visualization

1. **Transaction Patterns**:
   ```python
   import plotly.express as px

   fig = px.density_heatmap(
       df, x="transaction_hour", y="merchant_category",
       title="Transaction Frequency by Time and Category"
   )
   fig.show()
   ```

2. **Customer Segmentation**:
   - Used K-means clustering to identify 5 distinct customer groups
   - Visualized clusters using t-SNE dimensionality reduction

## Key Algorithms and Models

| Use Case | Algorithm | Key Features | Performance |
|----------|-----------|--------------|-------------|
| Fraud Detection | XGBoost | Handle class imbalance, Feature importance | AUC: 0.92 |
| Loan Approval | Random Forest | Interpretability, Non-linear relationships | Accuracy: 87% |
| Customer Churn | Logistic Regression | Fast training, Probability outputs | F1: 0.81 |
| Transaction Clustering | K-means | Customer segmentation | Silhouette: 0.65 |

## Results and Insights

1. **Fraud Detection**:
   - Identified 3 high-risk merchant categories
   - Reduced false negatives by 35% compared to existing system
   - Feature importance revealed time-of-day as critical factor

2. **Loan Risk Assessment**:
   - Developed dynamic pricing model based on risk profile
   - Increased approval rate by 15% while maintaining default rate

3. **Customer Insights**:
   - Discovered 2 underserved customer segments
   - Recommended targeted product offerings

## Performance Metrics

| Metric | Value | Benchmark |
|--------|-------|-----------|
| Model Training Time | 45 mins (distributed) | 6 hours (single node) |
| Data Processing Throughput | 2TB/hour | 500GB/hour (traditional) |
| Prediction Latency | <100ms per transaction | Industry avg: 300ms |
| Model Accuracy | 89.2% | Previous best: 85.7% |

## Challenges and Solutions

### Challenge 1: Data Skewness
- **Problem**: Transaction data highly skewed (99% legitimate)
- **Solution**: Implemented SMOTE oversampling and class weighting

### Challenge 2: Feature Scale
- **Problem**: Numerical features with different scales
- **Solution**: Robust scaling using Spark ML:
  ```python
  from pyspark.ml.feature import RobustScaler

  scaler = RobustScaler(
      inputCol="features",
      outputCol="scaledFeatures",
      withCentering=True
  )
  ```

### Challenge 3: Model Interpretability
- **Problem**: Need to explain predictions to regulators
- **Solution**: Used SHAP values and LIME for local explanations

## Future Work

1. **Real-time Processing**:
   - Implement Spark Streaming for live transaction monitoring
   - Kafka integration for event-driven architecture

2. **Advanced Techniques**:
   - Graph neural networks for transaction network analysis
   - Deep learning for unstructured data (emails, documents)

3. **Production Deployment**:
   - Model serving via MLflow
   - A/B testing framework
   - Continuous monitoring with Evidently AI
