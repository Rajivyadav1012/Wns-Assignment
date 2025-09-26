# Customer Segmentation Project

## Project Overview
The goal of this project is to cluster customers of a wholesale distributor based on their spending behavior across various product categories. By identifying distinct customer segments, businesses can improve marketing strategies, personalize offers, and enhance revenue generation.

- **Problem Type:** Unsupervised Learning (Clustering)  
- **Objective:** Identify customer segments and generate business insights for targeted marketing and resource allocation.

## Dataset
- **Source:** Wholesale Customers Dataset (Public)  
- **Shape:** 440 rows, 8 columns  
- **Features:**
  - `Channel`: Customer channel (dropped for clustering)
  - `Region`: Customer region (dropped for clustering)
  - `Fresh`, `Milk`, `Grocery`, `Frozen`, `Detergents_Paper`, `Delicassen`: Annual spending in monetary units  
- **Missing Values:** None  
- **Duplicates:** None  

## Data Preprocessing
1. Dropped categorical features `Channel` and `Region`.  
2. Checked for missing values and duplicates.  
3. Applied log transformation to skewed numerical features to reduce skewness.  
4. Scaled features using:  
   - **Standard Scaler** (for analysis)  
   - **Min-Max Scaler** (used for final modeling)  

## Exploratory Data Analysis (EDA)
- **Histograms:** To understand the distribution of spending per category.  
- **Boxplots:** For outlier detection.  
- **Correlation Analysis:** Heatmap to identify relationships between features.  
- **KDE Plots:** Compared original, Standard Scaled, and Min-Max Scaled data distributions.  

## Clustering Algorithms
1. **KMeans Clustering**  
   - Optimal clusters determined using Elbow Method and evaluation metrics.  
   - Tested cluster sizes from 2 to 9.  
   - Selected K=2 as the best cluster configuration.  
2. **Agglomerative Clustering**  
   - Hierarchical clustering used for comparison.  
3. **DBSCAN**  
   - Density-based clustering; did not find meaningful clusters with default parameters.  
4. **Gaussian Mixture Model (GMM)**  
   - Probabilistic clustering for comparison with KMeans.  

## Evaluation Metrics
- **Silhouette Score:** Measures cluster cohesion and separation (higher is better).  
- **Davies-Bouldin Index:** Measures similarity between clusters (lower is better).  
- **Calinski-Harabasz Index:** Measures variance ratio between and within clusters (higher is better).  

**Results Summary (Example for KMeans with K=2):**  
- Silhouette Score: 0.300  
- Davies-Bouldin Index: 1.309  
- Calinski-Harabasz Index: 200.53  

## Feature Engineering
- Total spending per customer calculated as the sum of all spending categories.  
- Clusters analyzed for dominant and weak categories to generate segment-specific insights.  

## Business Insights
- **Cluster 0: High-Value Grocery Focused**  
  - Size: 183 customers (41.6%)  
  - Value Tier: High Value  
  - Top Categories: Grocery, Milk, Fresh  
  - Weak Categories: Delicassen, Frozen  
  - Priority: High Priority  

- **Cluster 1: Fresh & Frozen Specialists**  
  - Size: 257 customers (58.4%)  
  - Value Tier: Medium Value  
  - Top Categories: Fresh, Frozen, Grocery  
  - Weak Categories: Detergents_Paper, Delicassen  
  - Priority: Medium Priority  

**Marketing Recommendations:**  
- Target cross-selling in weak categories.  
- Loyalty program enrollment for high-value customers.  
- Personalized campaigns for dominant spending categories.  

## Installation
1. Clone the repository:
```bash
git clone <repository-url>
cd customer-clustering
```

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:
```bash
.\venv\Scripts\Activate
```

4. Install project dependencies:
```bash
pip install -r requirements.txt
```

5. Verify installation:
```bash
python --version
pip list
```

6. Directory Structure after installation:
```
customer-clustering/
│
├─ data/                  # Dataset files
├─ models/                # Saved models and scalers
├─ notebooks/             # EDA and experimentation notebooks
├─ train.py               # Training and saving model/scaler
├─ predict.py             # Predict single customer cluster
├─ batch_predict.py       # Predict multiple customer clusters
├─ requirements.txt       # Python dependencies
├─ README.md              # Project documentation
```

## How to Run

### 1. Train the Model
Train the KMeans clustering model and save the scaler:
```bash
python train.py
```
- Saves `kmeans_model.pkl` and `minmax_scaler.pkl` to the `models/` folder.

### 2. Predict a Single Customer
Predict the cluster for a single customer:
```bash
python predict.py
```
- Input a dictionary with customer spending in categories: `['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']`.
- Outputs predicted cluster, segment, total spending, and recommendations.

### 3. Batch Prediction
Predict clusters for multiple customers:
```bash
python batch_predict.py
```
- Reads multiple customer records from `new_customers_sample.csv`.
- Saves predictions to `new_customers_sample_with_predictions.csv`.

### Notes
- Make sure `models/kmeans_model.pkl` and `models/minmax_scaler.pkl` exist before running predictions.
- Input values should be annual spending amounts for all categories.




