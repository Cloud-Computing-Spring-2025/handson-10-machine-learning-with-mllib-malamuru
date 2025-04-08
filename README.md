# handson-10-MachineLearning-with-MLlib.

# **Customer Churn Prediction with Spark MLlib**

This project uses **Apache Spark MLlib** to build, tune, and compare classification models for predicting customer churn. It includes data preprocessing, feature engineering, logistic regression training, chi-square feature selection, and model comparison with hyperparameter tuning.


## **Overview**

By analyzing customer subscription and usage data, the pipeline predicts whether a customer is likely to churn. The output includes performance metrics like AUC and identifies the best model for churn prediction.


## **Prerequisites**

Ensure the following are installed:

1. **Python 3.x**
   - [Download Python](https://www.python.org/downloads/)
   - Verify installation:
     ```bash
     python --version
     ```

2. **Apache Spark + PySpark**
   - Install PySpark via pip:
     ```bash
     pip install pyspark
     ```

3. **Dataset**
   - File: `customer_churn.csv`  
   - Place it in the project root directory.
     ```bash
      python dataset-generator.py
     ```

---

## **Dataset Columns**

| Column         | Type    | Description                              |
|----------------|---------|------------------------------------------|
| gender         | String  | Gender of the customer                   |
| SeniorCitizen  | Integer | 1 if senior, 0 otherwise                 |
| tenure         | Integer | Number of months with company            |
| PhoneService   | String  | Whether phone service is active          |
| InternetService| String  | Type of internet connection              |
| MonthlyCharges | Double  | Monthly bill                             |
| TotalCharges   | Double  | Total billed amount                      |
| Churn          | String  | Target label (Yes/No)                    |

---

## **Execution**

Run the complete pipeline using Spark:

```bash
spark-submit customer-churn-analysis.py
```

Output will be written to:

```bash
model_outputs.txt
```

---

## **Pipeline Tasks**

---

### **1. Data Preprocessing & Feature Engineering**

**Objective:**
- Handle missing values
- Encode categorical variables
- Assemble all features into a single vector

**Output:**
First 5 rows of `features` and `Index`:

-Data Preprocessing 
-Sample processed rows (features and Index):

| features                                      | ChurnIndex |
|---------------------------------------------- |------------|
|[0.0, 40.0, 31.08, 1211.1, 0.0, 1.0, 0.0, 1.0] |      0.0   |
|[8, {1: 63.0, 2: 64.51, 3: 3929.71, 6: 1.0}]   |      1.0   |
|[8, {0: 1.0, 1: 2.0, 2: 100.09, 5: 1.0}]       |      0.0   |
|[1.0, 39.0, 72.02, 3178.37, 0.0, 1.0, 1.0, 0.0]|      0.0   |
|[8, {1: 11.0, 2: 76.74, 3: 898.39}]            |      0.0   |


---

### **2. Train and Evaluate Logistic Regression**

**Objective:**
Train a logistic regression model and compute AUC.

**Output:**
```
-Logistic Regression 
Logistic Regression Model Accuracy : 0.7094
```

---

### **3. Feature Selection (Chi-Square Test)**

**Objective:**
Select the 5 most important features.

**Output Example:**


Top 5 selected features (first 5 rows):

| features                  | ChurnIndex |
|-------------------------- |------------|
|[0.0, 40.0, 0.0, 0.0, 1.0] |      0.0   |
|5, {1: 63.0, 3: 1.0}       |      1.0   |
|5, {0: 1.0, 1: 2.0}        |      0.0   |
|[1.0, 39.0, 0.0, 1.0, 0.0] |      0.0   |
|5, {1: 11.0}               |      1.0   |


---

### **4. Hyperparameter Tuning & Model Comparison**

**Objective:**
Use `CrossValidator` to tune multiple models and compare AUC scores.

**Models Evaluated:**
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosted Trees (GBT)

**Output:**

```text
=== Model Tuning and Comparison ===
LogisticRegression AUC: 0.7154
DecisionTree AUC: 0.7069
RandomForest AUC: 0.7490
GBTClassifier AUC: 0.7854
Best model: GBTClassifier with AUC = 0.7854
```

---

## **Output Format**

```bash
model_outputs.txt
```

This file logs:
- Preprocessing samples
- AUC of Logistic Regression
- Top 5 Chi-Square features
- AUC of all models and the best performer

![image](https://github.com/user-attachments/assets/f9ac6eee-da0d-426e-b5e5-e95cffe93c49)


---

## **Conclusion**

This assignment demonstrates how to build and optimize an end-to-end machine learning pipeline using PySpark. It reinforces skills in data engineering, model evaluation, and feature selection on distributed data systems.

---

