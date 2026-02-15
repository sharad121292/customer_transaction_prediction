Customer Transaction Prediction
📌 Overview

This project focuses on predicting the likelihood of future customer transactions using high-dimensional transactional data. The objective is to build a robust machine learning classification model capable of handling class imbalance and feature complexity while optimizing predictive performance using ROC-AUC as the primary evaluation metric.

The project demonstrates end-to-end implementation of data preprocessing, feature engineering, dimensionality reduction, model training, and evaluation.
-- 

📊 Dataset

High-dimensional anonymized transactional dataset

Binary target variable (Transaction: 0 = No, 1 = Yes)

Contains numerous numerical features

Class imbalance observed in target distribution

No direct business-identifiable information (privacy-safe dataset)

🛠 Tools & Technologies

Python

Pandas

NumPy

Scikit-learn

XGBoost

PCA (Principal Component Analysis)

Matplotlib

Seaborn

🔄 Project Steps
1️⃣ Data Preprocessing

Checked missing values

Handled class imbalance

Standardized numerical features

Removed potential data leakage

2️⃣ Exploratory Data Analysis (EDA)

Distribution analysis

Correlation checks

Target imbalance visualization

3️⃣ Feature Engineering

Applied feature scaling

Used PCA for dimensionality reduction

Reduced noise in high-dimensional data

4️⃣ Model Building

Trained and evaluated multiple models:

Logistic Regression

Random Forest

XGBoost

5️⃣ Model Evaluation

Primary Metric: ROC-AUC

Secondary Metrics: Accuracy, Precision, Recall, F1-score

Cross-validation for robustness

📈 Key Results & Insights

XGBoost delivered the best ROC-AUC performance.

PCA improved computational efficiency without major loss of predictive power.

Handling class imbalance significantly improved recall.

Feature reduction helped minimize overfitting.

Model demonstrates strong capability in predicting rare transaction events.  

▶️ How to Run the Project
Step 1: Clone the Repository
git clone https://github.com/yourusername/customer_transaction_prediction.git
cd customer_transaction_prediction

Step 2: Create Virtual Environment (Optional but Recommended)
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

Step 3: Install Dependencies
pip install -r requirements.txt


If requirements.txt is not available:

pip install pandas numpy scikit-learn xgboost matplotlib seaborn

Step 4: Run Jupyter Notebook
jupyter notebook


Open:

customer_transaction_prediction_done.ipynb


Run all cells to reproduce results.

📜 License

This project is licensed under the MIT License.
