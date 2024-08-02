# Lead-Score-Prediction-Using-Logistic-Regression
This repository includes a Python notebook, presentation, and summary detailing the process of predicting lead scores using logistic regression. The project covers data cleaning, EDA, feature engineering, model building, and evaluation. It provides a summary and answers to subjective questions related to the project. The final model helps identify potential leads likely to convert, aiding targeted marketing efforts.

# Dataset
The dataset includes 100,798 entries with the following columns:

S.No.: Serial number (integer)
Year: Year of the incident (integer)
Month: Month of the incident (integer)
Date: Date of the incident (string)
Reason: Reason for the incident (string)
Education: Education level of the individual involved (string, has missing values)
Sex: Sex of the individual involved (string)
Age: Age of the individual involved (float, has missing values)
Race: Race of the individual involved (string)
Hispanic: Hispanic status coded as integers
Place of incident: Location of the incident (string, has missing values)
Police involvement: Whether police were involved, coded as 0 (No) or 1 (Yes)

# Project Workflow
**1. Data Understanding and Collection**
Initial overview of the dataset and its features.
**2. Data Cleaning**
Handling missing values.
Converting categorical variables to numerical using dummy encoding.
**3. Exploratory Data Analysis (EDA)**
Univariate Analysis: Examining individual variables' distribution.
Bivariate Analysis: Exploring relationships between pairs of variables using correlation and scatter plots.
Multivariate Analysis: Analyzing interactions between multiple variables using pair plots, heatmaps, and PCA.
**4. Feature Engineering**
Scaling features using StandardScaler.
Feature selection using Recursive Feature Elimination (RFE).
**5. Model Building**
Building a logistic regression model to predict lead scores.
Training the model on the training set.
**6. Model Evaluation**
Evaluating the model using accuracy, precision, recall, F1-score, ROC AUC, and confusion matrix.
Plotting ROC and Precision-Recall curves.
Residual analysis to check model fit.
**7. Hypothesis Testing**
Performing hypothesis testing to validate the significance of features.

# Code
The repository includes Jupyter notebooks with step-by-step code for each stage of the analysis and model building process.

# Results
The final model provides a lead score that helps in identifying potential leads with a higher likelihood of conversion. The model evaluation metrics indicate that the model is reliable and performs well on the test data.

# Repository Structure
-  data/: Contains the dataset.
-  notebooks/: Jupyter notebooks with detailed analysis and model building steps.
-  data_dictionary/: Detailed description of the dataset's features.
-  models/: Serialized model files.
-  reports/: Generated reports and visualizations.
-  summary/: Summary of the project findings.
-  presentation/: Presentation slides detailing the project.
-  subjective_questions/: Answers to subjective questions related to the project.
-  README.md: Project overview and description.
