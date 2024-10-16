Team members
1. Muhammet Murat Albaş 
2. Ahmed Alkhulaifi
3.Tevfik Aybars Aydoğ

Project Title: Loan Eligibility Prediction

1. Objective of the Project: The objective is to create a machine learning model for loan approval
prediction.

2. Problem Statement: A loan company aims to automate the loan eligibility process. It uses
applicants' financial records and related information to determine the eligibility of individuals or
organizations for obtaining loans.

3. Dataset Details:
• Dataset Name: Loan approval dataset
• Source: Kaggle (https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset)
• Size: It has 4269 instances, 13 attributes, and the target variable is loan_status.
• Description: The dataset consists of 12 features, 4 of them are categorical features. The
dataset consists of the financial records of individuals and organizations. Like residential
assets value and commercial assets value. which will be used along with other associated
information, to determine the loan_status variable for applicants (Approved/Rejected).

4. Data Preprocessing:
• Handling Missing Values: There are no missing values in the dataset.
• Encoding Categorical Variables: Label Encoding was used on 3 categorical features.
• Feature Scaling/Normalization: standardization was applied on non-categorical features
• Exploratory Data Analysis: firstly, dataset.shape was used to know how many rows and
columns there are in the dataset. Then dataset.info() was used to know how many
instances there are in the dataset, and the variables types. Also we checked if the dataset
is balanced by plotting the target variable.

5. Machine Learning Models Used:
• Model 1: K-Nearest Neighbors
Justification: Knn algorithm is simple to implement and understand. It works well with
small datasets. Also, the Knn algorithm does not spend time during the training phase.
• Model 2: Random Forest Classification
Justification: The main reason to choose Random Forest algorithm is high accuracy. Also,
Random Forest Algorithm robust to overfitting.
• Model 3: Support Vector Machines
Justification: Svm handles outliers well. Svm robust to overfitting. Also, Svm works well
with a large number of features.

6. Hyperparameter Tuning:
• Model 1: Randomized Search applied. For n_neighbors parameter ‘11’, for metrics
‘euclidean’, for p ‘1’, weights ‘distance’ we got the best accuracy.
• Model 2: Randomized Search applied. For n_estimators ‘50’, for min_samples_split
‘10’, for min_samples_leaf ‘1’, for max_depth ‘10’, for bootstrap ‘true’ we got the best
accuracy.
• Model 3: Randomized Search applied. For kernel ‘rbf’, for c ‘1’, for gamma ‘0.1’, for
degree ‘3’, for coef0 ‘0.0’ we got the best accuracy.

7. Results:
Performance Metrics:We used accuracy, precision, recall, F1 score and support for evaluation
metrics.
Feature Selection Impact: Before feature selection accuracies are; knn:0.58, random forest:0.98,
Svm:0.61. After feature selection accuracies are; knn:0.91, random forest:0.95, svm:0.92.
Insights and Observations: From the analysis, random forest has the highest accuracy then svm and
knn. After hyperparameter tuning we got improvement on precision, recall and F1-score for knn but it
stayed the same for random forest algorithm and svm.

8. Conclusion: As a result, we got the best results for the loan approval dataset with random
forest algorithm. So, we observed the power of random forest algorithm again. And for
hyperparameter tuning we understood that maybe we should use some other techniques in the
future to get better improvements on the results.
