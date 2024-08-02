#!/usr/bin/env python
# coding: utf-8

# # Lead Score - Case Study
# 
# ## Problem Statement
# 
# An X Education need help to select the most promising leads, i.e. the leads that are most likely to convert into paying customers. The company requires us to build a model wherein you need to assign a lead score to each of the leads such that the customers with higher lead score have a higher conversion chance and the customers with lower lead score have a lower conversion chance. The CEO, in particular, has given a ballpark of the target lead conversion rate to be around 80%.

# ## Goals and Objectives
# There are quite a few goals for this case study.
# 
# -  Build a logistic regression model to assign a lead score between 0 and 100 to each of the leads which can be used by the company to target potential leads. A higher score would mean that the lead is hot, i.e. is most likely to convert whereas a lower score would mean that the lead is cold and will mostly not get converted.
# 

# ## 1. Importing Libries

# In[1]:


# Filtering out the warnings

import warnings

warnings.filterwarnings('ignore')


# In[2]:


import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, classification_report, precision_recall_curve
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
import statsmodels.api as sm
from sklearn.feature_selection import RFE
from statsmodels.stats.outliers_influence import variance_inflation_factor


# ## 2. Data Understanding and Collection:
# -  Load the data.
# -  Understand the data structure, types, and formats.

# In[3]:


# Load the data
file_path = r'C:\Users\Jyoti Mishra\Desktop\Leads.csv'
leads_df = pd.read_csv(file_path)

# Display the first few rows of the dataframe
leads_df.head()


# In[4]:


leads_df.info()


# In[5]:


leads_df.describe()


# In[6]:


leads_df.shape


# **The dataset contains 9240 entries with 37 columns.**

# ## 3. Data Preprocessing:
# -  Handle missing values.
# -  Remove duplicates.
# -  Handle outliers.

# In[7]:


# Checking for missing values in the dataset
missing_values = leads_df.isnull().sum()

# Display columns with missing values
missing_values[missing_values > 0]


# -  Drop columns with more than 40% missing values.
# -  Impute missing values for numerical columns using median.
# -  Impute missing values for categorical columns using mode.

# In[8]:


# Converting all the values to lower case
leads_df = leads_df.applymap(lambda s:s.lower() if type(s) == str else s)


# In[9]:


# Remove duplicates, if any
leads_df = leads_df.drop_duplicates()


# In[10]:


# Replacing 'Select' with NaN (Since it means no option is selected)
leads_df = leads_df.replace('select',np.nan)


# In[11]:


# Dropping columns with more than 40% missing values
threshold = len(leads_df) * 0.4
leads_df = leads_df.dropna(thresh=threshold, axis=1)

# List of remaining columns after dropping
remaining_columns = leads_df.columns

# Identifying numerical and categorical columns
numerical_cols = leads_df.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = leads_df.select_dtypes(include=['object']).columns

# Imputing missing values for numerical columns using median
leads_df[numerical_cols] = leads_df[numerical_cols].fillna(leads_df[numerical_cols].median())

# Imputing missing values for categorical columns using mode
for col in categorical_cols:
    leads_df[col] = leads_df[col].fillna(leads_df[col].mode()[0])

# Dropping any rows that still contain missing values
leads_df = leads_df.dropna()

# Verify that there are no more missing values
missing_values_after = leads_df.isnull().sum().sum()

missing_values_after, leads_df.shape


# In[12]:


# Identify outliers using IQR method
def treat_outliers(col):
    Q1 = leads_df[col].quantile(0.25)
    Q3 = leads_df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    leads_df[col] = leads_df[col].clip(lower=lower_bound, upper=upper_bound)

# Apply outlier treatment to numerical columns
for col in numerical_cols:
    treat_outliers(col)


# In[13]:


leads_df.describe()


# In[14]:


# Checking if there are columns with one unique value since it won't affect our analysis
leads_df.nunique()


# In[15]:


# Finding the null percentages across columns
round(leads_df.isnull().sum()/len(leads_df.index),2)*100


# In[16]:


leads_df["Country"].value_counts()


# In[17]:


def slots(x):
    category = ""
    if x == "india":
        category = "india"
    else:
        category = "outside india"
    return category

leads_df['Country'] = leads_df.apply(lambda x:slots(x['Country']), axis = 1)
leads_df['Country'].value_counts()


# In[18]:


leads_df['Country'].head()


# In[19]:


# To familiarize all the categorical values
for column in leads_df:
    print(leads_df[column].astype('category').value_counts())
    print('----------------------------------------------------------------------------------------')


# ## 4. Exploratory Data Analysis (EDA)
# ### 4.1 UNIVARIATE AND BIVARIATE

# -  **a.) Converted**
# 
# Converted is the target variable, Indicates whether a lead has been successfully converted (1) or not (0)
# 

# In[20]:


Converted = (sum(leads_df['Converted'])/len(leads_df['Converted'].index))*100
Converted


# **The lead conversion rate is 39%.**

# -  **b.) Lead Origin**

# In[21]:


plt.figure(figsize=(10,5))
sns.countplot(x = "Lead Origin", hue = "Converted", data = leads_df,palette='Set1')
plt.xticks(rotation = 45)


# **Inference :**
# -  API and Landing Page Submission have 30-35% conversion rate but count of lead originated from them are considerable.
# -  Lead Add Form has more than 90% conversion rate but count of lead are not very high.
# -  Lead Import and quick add form are very less in count.
# 
# **To improve overall lead conversion rate, we need to focus more on improving lead converion of API and Landing Page Submission origin and generate more leads from Lead Add Form.**

# -  **c.) Lead Source**

# In[22]:


plt.figure(figsize=(13,5))
sns.countplot(x = "Lead Source", hue = "Converted", data = leads_df, palette='Set1')
plt.xticks(rotation = 90)


# **Inference**
# -  Google and Direct traffic generates maximum number of leads.
# -  Conversion Rate of reference leads and leads through welingak website is high.
# 
# **To improve overall lead conversion rate, focus should be on improving lead converion of olark chat, organic search, direct traffic, and google leads and generate more leads from reference and welingak website.**
# 
# 

# **d.)Do not Email**
# 

# In[23]:


sns.countplot(x = "Do Not Email", hue = "Converted", data = leads_df,palette='Set1')
plt.xticks(rotation = 90)


# **Inference:**
# Most entries are 'No'. No Inference can be drawn with this parameter.

# **e.) Do not call**

# In[24]:


sns.countplot(x = "Do Not Call", hue = "Converted", data = leads_df,palette='Set1')
plt.xticks(rotation = 90)


# **Inference:**
# Most entries are 'No'. No Inference can be drawn with this parameter.

# **f.) TotalVisits**

# In[25]:


leads_df['TotalVisits'].describe(percentiles=[0.05,.25, .5, .75, .90, .95, .99])


# In[26]:


sns.boxplot(leads_df['TotalVisits'],orient='vert',palette='Set1')


# In[27]:


sns.boxplot(y = 'TotalVisits', x = 'Converted', data = leads_df,palette='Set1')


# **Inference:**
# -  Median for converted and not converted leads are the same.
# 
# **Nothing can be concluded on the basis of Total Visits.**

# **g.) Total Time Spent on Website**

# In[28]:


leads_df['Total Time Spent on Website'].describe()


# In[29]:


sns.boxplot(leads_df['Total Time Spent on Website'],orient='vert',palette='Set1')


# In[30]:


sns.boxplot(y = 'Total Time Spent on Website', x = 'Converted', data = leads_df,palette='Set1')


# **Inference:**
# -  Leads spending more time on the weblise are more likely to be converted.
# 
# **Website should be made more engaging to make leads spend more time.**

# **h.) Page Views Per Visit**

# In[31]:


leads_df['Page Views Per Visit'].describe()


# In[32]:


sns.boxplot(leads_df['Page Views Per Visit'],orient='vert',palette='Set1')


# In[33]:


sns.boxplot(y = 'Page Views Per Visit', x = 'Converted', data =leads_df,palette='Set1')


# **Inference:**
# -  Median for converted and unconverted leads is the same.
# 
# **Nothing can be said specifically for lead conversion from Page Views Per Visit**

# **i.)Last Activity**
# 

# In[34]:


leads_df['Last Activity'].describe()


# In[35]:


plt.figure(figsize=(15,6))
sns.countplot(x = "Last Activity", hue = "Converted", data = leads_df,palette='Set1')
plt.xticks(rotation = 90)


# **Inference:**
# -  Most of the lead have their Email opened as their last activity.
# 
# **Conversion rate for leads with last activity as SMS Sent is almost 60%.**

# **j.) Country**

# In[36]:


plt.figure(figsize=(15,6))
sns.countplot(x = "Country", hue = "Converted", data = leads_df,palette='Set1')
plt.xticks(rotation = 90)


# **Inference:** Most values are 'India' no such inference can be drawn

# **k.) Specialization**

# In[37]:


plt.figure(figsize=(15,6))
sns.countplot(x = "Specialization", hue = "Converted", data = leads_df,palette='Set1')
plt.xticks(rotation = 90)


# **Inference:**
# Focus should be more on the Specialization with high conversion rate.

# **l.) What is your current occupation**

# In[38]:


plt.figure(figsize=(15,6))
sns.countplot(x = "What is your current occupation", hue = "Converted", data = leads_df,palette='Set1')
plt.xticks(rotation = 90)


# **Inference:**
# -  Working Professionals going for the course have high chances of joining it.
# 
# **Unemployed leads are the most in numbers but has around 30-35% conversion rate.**

# **m.) Search**

# In[39]:


sns.countplot(x = "Search", hue = "Converted", data = leads_df,palette='Set1')
plt.xticks(rotation = 90)


# **Inference:**
# Most entries are 'No'. No Inference can be drawn with this parameter.

# **n.)Magazine**

# In[40]:


sns.countplot(x = "Magazine", hue = "Converted", data = leads_df,palette='Set1')
plt.xticks(rotation = 90)


# **Inference:**
# Most entries are 'No'. No Inference can be drawn with this parameter.

# **o.) Newspaper Article**
# 

# In[41]:


sns.countplot(x = "Newspaper Article", hue = "Converted", data = leads_df,palette='Set1')
plt.xticks(rotation = 90)


# **Inference:**
# Most entries are 'No'. No Inference can be drawn with this parameter.

# **p.) X Education Forums**
# 

# In[42]:


sns.countplot(x = "X Education Forums", hue = "Converted", data = leads_df,palette='Set1')
plt.xticks(rotation = 90)


# **Inference:**
# Most entries are 'No'. No Inference can be drawn with this parameter.

# **q.) Newspaper**

# In[43]:


sns.countplot(x = "Newspaper", hue = "Converted", data = leads_df,palette='Set1')
plt.xticks(rotation = 90)


# **Inference:**
# Most entries are 'No'. No Inference can be drawn with this parameter.

# **r.) Digital Advertisement**
# 

# In[44]:


sns.countplot(x = "Digital Advertisement", hue = "Converted", data = leads_df,palette='Set1')
plt.xticks(rotation = 90)


# **Inference:**
# Most entries are 'No'. No Inference can be drawn with this parameter.

# **s.)Through Recommendations**

# In[45]:


sns.countplot(x = "Through Recommendations", hue = "Converted", data = leads_df,palette='Set1')
plt.xticks(rotation = 90)


# **Inference:**
# Most entries are 'No'. No Inference can be drawn with this parameter.

# **t.) Receive More Updates About Our Courses**

# In[46]:


sns.countplot(x = "Receive More Updates About Our Courses", hue = "Converted", data = leads_df,palette='Set1')
plt.xticks(rotation = 90)


# **Inference:**
# Most entries are 'No'. No Inference can be drawn with this parameter.

# **u.) Tags**

# In[47]:


plt.figure(figsize=(15,6))
sns.countplot(x = "Tags", hue = "Converted", data = leads_df,palette='Set1')
plt.xticks(rotation = 90)


# **Inference:**
# -  Since this is a column which is generated by the sales team for their analysis , so this is not available for model 
# 
# **building . So we will need to remove this column before building the model.**

# **v.)City**

# In[48]:


plt.figure(figsize=(15,5))
sns.countplot(x = "City", hue = "Converted", data = leads_df,palette='Set1')
plt.xticks(rotation = 90)


# **Inference:**
# Most leads are from mumbai with around 50% conversion rate.

# **w.)Last Notable Activity**

# In[49]:


plt.figure(figsize=(15,5))
sns.countplot(x = "Last Notable Activity", hue = "Converted", data = leads_df,palette='Set1')
plt.xticks(rotation = 90)


# In[50]:


# Bivariate analysis for numerical vs. target variable
num_numerical_cols = len(numerical_cols)
n_rows = (num_numerical_cols // 4) + int(num_numerical_cols % 4 != 0)  # Calculate rows needed

plt.figure(figsize=(16, 4 * n_rows))
for i, col in enumerate(numerical_cols):
    plt.subplot(n_rows, 4, i + 1)
    sns.boxplot(x='Converted', y=col, data=leads_df)
    plt.title(f'{col} vs. Converted')
plt.tight_layout()
plt.show()


# **Inference:** 
# -  **Constant Variables (Asymmetrique Activity Score and Asymmetrique Profile Score):**
# 
# No Discriminatory Power: Since these scores are constants, they do not vary between different leads. Therefore, they do not have any discriminatory power to predict or explain conversion. They can be considered for removal from the dataset as they do not add value to the predictive model or analysis.
# 
# -  **Total Visits vs. Converted:**
# 
# The box plot shows that leads that have converted tend to have higher Total Visits. This suggests that leads with more interactions (higher total visits) are more likely to convert.
# 
# -  **Total Time Spent on Website vs. Converted:**
# 
# Converted leads tend to spend more time on the website. This indicates that engagement time is a positive indicator of conversion likelihood.
# 
# -  **Page Views Per Visit vs. Converted:**
# 
# There seems to be a slightly higher number of page views per visit for converted leads. This can suggest that more engaged users who explore more pages per visit are more likely to convert.

# ### 4.2 MULTIVARIATE ANALYSIS

# In[51]:


# Pair plots for numerical variables
sns.pairplot(leads_df[numerical_cols], diag_kind='kde', hue='Converted')
plt.show()


# **Inference:**
# -  **Total Visits:**
# 
# Converted vs. Non-Converted: The scatter plots and density plots indicate that both converted and non-converted leads have a wide range of total visits. However, there seems to be a slight concentration of converted leads with higher total visits, suggesting that leads who visit the website more frequently have a higher chance of conversion.
# 
# -  **Total Time Spent on Website:**
# 
# Engagement Indicator: Converted leads tend to spend more time on the website. The density plot shows a higher concentration of converted leads in the upper range of total time spent, indicating that time spent on the website is a strong indicator of conversion likelihood.
# 
# -  **Page Views Per Visit:**
# 
# Exploration Behavior: Converted leads generally have a higher number of page views per visit compared to non-converted leads. This suggests that leads who explore more pages during their visits are more engaged and have a higher probability of converting.
# 
# -  **Asymmetrique Activity Score and Asymmetrique Profile Score:**
# 
# Constant Values: Since these scores are constants (14 for Activity Score and 16 for Profile Score), they do not provide any meaningful variance or insight when comparing converted and non-converted leads. Therefore, they do not contribute to distinguishing between the two groups.

# ## 5. Hypothesis Testing

# In[52]:


# Hypothesis 1: Total Time Spent on Website
converted = leads_df[leads_df['Converted'] == 1]['Total Time Spent on Website']
not_converted = leads_df[leads_df['Converted'] == 0]['Total Time Spent on Website']
t_stat, p_val = stats.ttest_ind(converted, not_converted)
print(f"Hypothesis 1: p-value = {p_val}")
if p_val < 0.05:
    print("Reject the null hypothesis: There is a significant difference in the total time spent on the website between leads that converted and those that did not.")
else:
    print("Fail to reject the null hypothesis: There is no significant difference in the total time spent on the website between leads that converted and those that did not.")


# In[53]:


# Hypothesis 2: Lead Source and Conversion
contingency_table = pd.crosstab(leads_df['Lead Source'], leads_df['Converted'])
chi2_stat, p_val, dof, ex = stats.chi2_contingency(contingency_table)
print(f"\nHypothesis 2: p-value = {p_val}")
if p_val < 0.05:
    print("Reject the null hypothesis: There is a significant association between the lead source and whether the lead converted or not.")
else:
    print("Fail to reject the null hypothesis: There is no significant association between the lead source and whether the lead converted or not.")


# In[54]:


# Hypothesis 3: Total Visits
converted = leads_df[leads_df['Converted'] == 1]['TotalVisits']
not_converted = leads_df[leads_df['Converted'] == 0]['TotalVisits']
t_stat, p_val = stats.ttest_ind(converted, not_converted)
print(f"\nHypothesis 3: p-value = {p_val}")
if p_val < 0.05:
    print("Reject the null hypothesis: There is a significant difference in the number of total visits between leads that converted and those that did not.")
else:
    print("Fail to reject the null hypothesis: There is no significant difference in the number of total visits between leads that converted and those that did not.")


# In[55]:


# Hypothesis 4: Page Views Per Visit
converted = leads_df[leads_df['Converted'] == 1]['Page Views Per Visit']
not_converted = leads_df[leads_df['Converted'] == 0]['Page Views Per Visit']
t_stat, p_val = stats.ttest_ind(converted, not_converted)
print(f"\nHypothesis 4: p-value = {p_val}")
if p_val < 0.05:
    print("Reject the null hypothesis: There is a significant difference in the page views per visit between leads that converted and those that did not.")
else:
    print("Fail to reject the null hypothesis: There is no significant difference in the page views per visit between leads that converted and those that did not.")


# In[56]:


# Hypothesis 5: City and Conversion
contingency_table = pd.crosstab(leads_df['City'], leads_df['Converted'])
chi2_stat, p_val, dof, ex = stats.chi2_contingency(contingency_table)
print(f"\nHypothesis 5: p-value = {p_val}")
if p_val < 0.05:
    print("Reject the null hypothesis: There is a significant association between the city of the lead and whether the lead converted or not.")
else:
    print("Fail to reject the null hypothesis: There is no significant association between the city of the lead and whether the lead converted or not.")


# In[57]:


# Hypothesis 6: Asymmetrique Activity Score
converted = leads_df[leads_df['Converted'] == 1]['Asymmetrique Activity Score']
not_converted = leads_df[leads_df['Converted'] == 0]['Asymmetrique Activity Score']
t_stat, p_val = stats.ttest_ind(converted, not_converted)
print(f"\nHypothesis 6: p-value = {p_val}")
if p_val < 0.05:
    print("Reject the null hypothesis: There is a significant difference in the Asymmetrique Activity Score between leads that converted and those that did not.")
else:
    print("Fail to reject the null hypothesis: There is no significant difference in the Asymmetrique Activity Score between leads that converted and those that did not.")


# **Results : Based on the EDA we have seen that many columns are not adding any information to the model, hence we can drop them for further analysis**

# In[58]:


leads_df = leads_df.drop(['Lead Number','Tags','Country','Search','Magazine','Newspaper Article',
                          'X Education Forums','Newspaper','Digital Advertisement',
                          'Through Recommendations','Receive More Updates About Our Courses',
                          'Update me on Supply Chain Content','Get updates on DM Content',
                          'I agree to pay the amount through cheque','A free copy of Mastering The Interview','Asymmetrique Activity Score','Asymmetrique Profile Score','Prospect ID'],
                          axis=1)


# In[59]:


leads_df.shape


# In[60]:


leads_df.info()


# ## 6. Creating Dummy Variables

# In[61]:


#Converting some binary variables (Yes/No) to 1/0
vars =  ['Do Not Email', 'Do Not Call']

def binary_map(x):
    return x.map({'yes': 1, "no": 0})

leads_df[vars] = leads_df[vars].apply(binary_map)


# In[62]:


#Creating Dummy Variables
leads_df = pd.get_dummies(leads_df, drop_first=True)


# In[63]:


# Convert boolean columns to integers (0 and 1)
bool_cols = leads_df.select_dtypes(include=['bool']).columns
leads_df[bool_cols] = leads_df[bool_cols].astype(int)

# Check the data types of your columns
print(leads_df.dtypes)


# ## 7. Splitting Data into Train and Test Sets

# In[64]:


X = leads_df.drop('Converted', axis=1)
y = leads_df['Converted']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# ## 8. Scaling the Features

# In[65]:


scaler = StandardScaler()
X_train[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']] = scaler.fit_transform(X_train[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']])

X_train.head()


# In[66]:


# Checking the Lead Conversion rate
Converted = (sum(leads_df['Converted'])/len(leads_df['Converted'].index))*100
Converted


# **We have almost 38.5% lead conversion rate.**

# ## 9. Feature Selection Using RFE

# In[67]:


logreg = LogisticRegression()

rfe = RFE(estimator=logreg, n_features_to_select=20)         # running RFE with 20 variables as output
rfe = rfe.fit(X_train, y_train)


# In[68]:


rfe.support_


# In[69]:


list(zip(X_train.columns, rfe.support_, rfe.ranking_))


# In[70]:


cols = X_train.columns[rfe.support_]
cols


# ## 10. Model Building

# ### Model-1

# -  **Initial Model Training**

# In[71]:


X_train_sm = sm.add_constant(X_train[cols])
logm1 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
result = logm1.fit()
result.summary()


# 
# **Since Pvalue of 'Lead Source_welingak website' is very high, we can drop this column.**

# -  **Model Evaluation**

# In[72]:


# Predictions using the model
y_train_pred_prob = result.predict(X_train_sm)
y_train_pred = np.where(y_train_pred_prob >= 0.5, 1, 0)


# In[73]:


# Accuracy
accuracy1 = accuracy_score(y_train, y_train_pred)
print(f'Accuracy: {accuracy1}')


# In[74]:


# Precision
precision1 = precision_score(y_train, y_train_pred)
print(f'Precision: {precision1}')


# In[75]:


# Recall
recall1 = recall_score(y_train, y_train_pred)
print(f'Recall: {recall1}')


# In[76]:


# F1 Score
f11 = f1_score(y_train, y_train_pred)
print(f'F1 Score: {f11}')


# In[77]:


# ROC AUC
roc_auc1 = roc_auc_score(y_train, y_train_pred_prob)
print(f'ROC AUC: {roc_auc1}')


# In[78]:


# Confusion Matrix
conf_matrix1 = confusion_matrix(y_train, y_train_pred)
print('Confusion Matrix:\n', conf_matrix1)


# In[79]:


# Classification Report
class_report = classification_report(y_train, y_train_pred)
print('Classification Report:\n', class_report)


# In[80]:


# ROC Curve
fpr, tpr, thresholds = roc_curve(y_train, y_train_pred_prob)
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'ROC Curve (area = {roc_auc1:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()


# In[81]:


# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_train, y_train_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label='Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()


# -  **Residual Analysis**

# In[82]:


# Calculate residuals
residuals = y_train - y_train_pred_prob

# Plot residuals distribution
sns.histplot(residuals, kde=True)
plt.title('Residuals Distribution')
plt.show()

# Q-Q Plot to check normality
sm.qqplot(residuals, line='45')
plt.title('Q-Q Plot of Residuals')
plt.show()


# ## Model-2

# In[83]:


# Dropping the column 'What is your current occupation_Housewife'
col1 = cols.drop('Lead Source_welingak website')


# -  **Initial Model Training**

# In[84]:


X_train_sm = sm.add_constant(X_train[col1])
logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()


# **Since Pvalue of 'Last Notable Activity_had a phone conversation' is very high, we can drop this column.**

# -  **Model Evaluation**

# In[85]:


# Model Predictions
y_train_pred_prob = res.predict(X_train_sm)
y_train_pred = np.where(y_train_pred_prob >= 0.5, 1, 0)


# In[86]:


# Accuracy
accuracy2 = accuracy_score(y_train, y_train_pred)
print(f'Accuracy: {accuracy2}')


# In[87]:


# Precision
precision2 = precision_score(y_train, y_train_pred)
print(f'Precision: {precision2}')


# In[88]:


#recall
recall2 = recall_score(y_train, y_train_pred)
print(f'recall: {recall2}')


# In[89]:


# F1 Score
f12 = f1_score(y_train, y_train_pred)
print(f'F1 Score: {f12}')


# In[90]:


# ROC AUC
roc_auc2 = roc_auc_score(y_train, y_train_pred_prob)
print(f'ROC AUC: {roc_auc2}')


# In[91]:


# Confusion Matrix
conf_matrix2 = confusion_matrix(y_train, y_train_pred)
print('Confusion Matrix:\n', conf_matrix2)


# In[92]:


# Classification Report
class_report = classification_report(y_train, y_train_pred)
print('Classification Report:\n', class_report)


# In[93]:


# ROC Curve
fpr, tpr, thresholds = roc_curve(y_train, y_train_pred_prob)
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'ROC Curve (area = {roc_auc2:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()


# In[94]:


# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_train, y_train_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label='Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()


# -  **Residual Analysis**

# In[95]:


# Calculate residuals
residuals = y_train - y_train_pred_prob

# Plot residuals distribution
sns.histplot(residuals, kde=True)
plt.title('Residuals Distribution')
plt.show()

# Q-Q Plot to check normality
sm.qqplot(residuals, line='45')
plt.title('Q-Q Plot of Residuals')
plt.show()


# ## Model-3

# In[96]:


col1 = col1.drop('Last Notable Activity_had a phone conversation')


# -  **Initial Model Training**

# In[97]:


X_train_sm = sm.add_constant(X_train[col1])
logm3 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm3.fit()
res.summary()


# **Checking for VIF values**

# In[98]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train[col1].columns
vif['VIF'] = [variance_inflation_factor(X_train[col1].values, i) for i in range(X_train[col1].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# -  **Model Evaluation**

# In[99]:


# Model Predictions
y_train_pred_prob = res.predict(X_train_sm)
y_train_pred = np.where(y_train_pred_prob >= 0.5, 1, 0)


# In[100]:


# Accuracy
accuracy3 = accuracy_score(y_train, y_train_pred)
print(f'Accuracy: {accuracy3}')


# In[101]:


# Precision
precision3 = precision_score(y_train, y_train_pred)
print(f'Precision: {precision3}')


# In[102]:


#recall
recall3 = recall_score(y_train, y_train_pred)
print(f'recall: {recall3}')


# In[103]:


# F1 Score
f13 = f1_score(y_train, y_train_pred)
print(f'F1 Score: {f13}')


# In[104]:


# ROC AUC
roc_auc3 = roc_auc_score(y_train, y_train_pred_prob)
print(f'ROC AUC: {roc_auc3}')


# In[105]:


# Confusion Matrix
conf_matrix3 = confusion_matrix(y_train, y_train_pred)
print('Confusion Matrix:\n', conf_matrix3)


# In[106]:


# Classification Report
class_report = classification_report(y_train, y_train_pred)
print('Classification Report:\n', class_report)


# In[107]:


# ROC Curve
fpr, tpr, thresholds = roc_curve(y_train, y_train_pred_prob)
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'ROC Curve (area = {roc_auc3:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()


# In[108]:


# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_train, y_train_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label='Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()


# -  **Residual Analysis**

# In[109]:


# Calculate residuals
residuals = y_train - y_train_pred_prob

# Plot residuals distribution
sns.histplot(residuals, kde=True)
plt.title('Residuals Distribution')
plt.show()

# Q-Q Plot to check normality
sm.qqplot(residuals, line='45')
plt.title('Q-Q Plot of Residuals')
plt.show()


# ## Model-4

# In[110]:


# Dropping the column  'What is your current occupation_Unemployed' because it has high VIF
col1 = col1.drop('What is your current occupation_unemployed')


# In[111]:


X_train_sm = sm.add_constant(X_train[col1])
logm5 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm5.fit()
res.summary()


# **Checking for VIF values**

# In[112]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train[col1].columns
vif['VIF'] = [variance_inflation_factor(X_train[col1].values, i) for i in range(X_train[col1].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# **Since the Pvalues of all variables is 0 and VIF values are low for all the variables, model-4 is our final model. We have 17 variables in our final model.**

# **Making Prediction on the Train set**

# In[113]:


# Getting the predicted values on the train set
y_train_pred = res.predict(X_train_sm)
y_train_pred[:10]


# In[114]:


#Creating a dataframe with the actual Converted flag and the predicted probabilities
y_train_pred_final = pd.DataFrame({'Converted':y_train.values, 'Converted_prob':y_train_pred})
y_train_pred_final['Prospect ID'] = y_train.index
y_train_pred_final.head()


# In[115]:


# Creating new column 'predicted' with 1 if Converted_Prob > 0.5 else 0
y_train_pred_final['predicted'] = y_train_pred_final.Converted_prob.map(lambda x: 1 if x > 0.5 else 0)

# Let's see the head
y_train_pred_final.tail()


# -  **Model Evaluation**

# In[116]:


# Confusion matrix 
confusion4 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.predicted )
print(confusion4)


# In[117]:


# Accuracy
accuracy4 = accuracy_score(y_train_pred_final.Converted, y_train_pred_final.predicted)
print(f'Accuracy: {accuracy4}')


# In[118]:


# Metrics beyond simply accuracy
TP = confusion4[1,1] # true positive 
TN = confusion4[0,0] # true negatives
FP = confusion4[0,1] # false positives
FN = confusion4[1,0] # false negatives


# In[119]:


# Sensitivity of our logistic regression model
print("Sensitivity : ",TP / float(TP+FN))


# In[120]:


# Let us calculate specificity
print("Specificity : ",TN / float(TN+FP))


# In[121]:


# Calculate false postive rate - predicting converted lead when the lead actually was not converted
print("False Positive Rate :",FP/ float(TN+FP))


# In[122]:


# positive predictive value 
print("Positive Predictive Value :",TP / float(TP+FP))


# In[123]:


# Negative predictive value
print ("Negative predictive value :",TN / float(TN+ FN))


# In[124]:


#precision
precision4 = precision_score(y_train_pred_final.Converted, y_train_pred_final.predicted)
print(f'Precision: {precision4}')


# In[125]:


#recall
recall4 = recall_score(y_train_pred_final.Converted, y_train_pred_final.predicted)
print(f'recall: {recall4}')


# In[126]:


# F1 Score
f14= f1_score(y_train_pred_final.Converted, y_train_pred_final.predicted)
print(f'F1 Score: {f14}')


# In[127]:


# ROC AUC
roc_auc4 = roc_auc_score(y_train_pred_final.Converted, y_train_pred_final.predicted)
print(f'ROC AUC: {roc_auc4}')


# In[128]:


# Classification Report
class_report = classification_report(y_train_pred_final.Converted, y_train_pred_final.predicted)
print('Classification Report:\n', class_report)


# In[129]:


# ROC Curve
fpr, tpr, thresholds = roc_curve(y_train_pred_final.Converted, y_train_pred_final.predicted)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (area = {roc_auc4:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.show()


# In[130]:


# Precision-Recall Curve
precision, recall, thresholds = precision_recall_curve(y_train_pred_final.Converted, y_train_pred_final.predicted)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='.')
plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.grid(True)
plt.show()


# -  **Residual Analysis**

# In[131]:


# Residuals: Assuming 'predicted_prob' column contains predicted probabilities
y_train_pred_final['residuals'] = y_train_pred_final['Converted'] - y_train_pred_final['predicted']


# In[132]:


# Plot residuals distribution
sns.histplot(y_train_pred_final['residuals'], kde=True)
plt.title('Residuals Distribution')
plt.show()


# In[133]:


# Q-Q Plot to check normality of residuals
sm.qqplot(y_train_pred_final['residuals'], line='45')
plt.title('Q-Q Plot of Residuals')
plt.show()


# In[135]:


# Residuals vs Fitted Values Plot
plt.figure(figsize=(8, 6))
plt.scatter(y_train_pred_final['predicted'], y_train_pred_final['residuals'], alpha=0.5)
plt.axhline(0, color='r', linestyle='--', linewidth=2)
plt.title('Residuals vs Fitted Values')
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.grid(True)
plt.show()


# In[140]:


# Create a dictionary to store the results
results = {
    "Metric": ["Confusion Matrix", "Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"],
    "Model 1": [conf_matrix1, accuracy1, precision1, recall1, f11, roc_auc1],
    "Model 2": [conf_matrix2, accuracy2, precision2, recall2, f12, roc_auc2],
    "Model 3": [conf_matrix3, accuracy3, precision3, recall3, f13, roc_auc3],
    "Model 4": [confusion4, accuracy4, precision4, recall4, f14, roc_auc4]}

# Convert the dictionary to a pandas DataFrame
results_df = pd.DataFrame(results)

# Display the results
print(results_df)


# In[141]:


# The confusion matrix indicates as below
# Predicted     not_converted    converted
# Actual
# not_converted        3583      401
# converted            584       1900 


# <B> <FONT COLOR='GREEN'>The model 4 is particularly effective in identifying potential conversions, as evidenced by its confusion matrix. Out of 2,484 actual converted leads, the model correctly identified 1,900, achieving a high sensitivity (Recall). This suggests that the model is well-suited for the task of focusing on leads likely to convert, allowing the team to prioritize efforts on these high-potential prospects. The relatively low number of false negatives (584) indicates that the model captures the majority of those who can be converted, making it the best among all models for this purpose. so we will use Model 4 </B> </FONT>

# ## 11. Finding Optimal Cutoff Point

# Above we had chosen an arbitrary cut-off value of 0.5. We need to determine the best cut-off value and the below section deals with that. Optimal cutoff probability is that prob where we get balanced sensitivity and specificity

# In[142]:


# Let's create columns with different probability cutoffs 
numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final.Converted_prob.map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()


# In[143]:


# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.
cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])
from sklearn.metrics import confusion_matrix

# TP = confusion[1,1] # true positive 
# TN = confusion[0,0] # true negatives
# FP = confusion[0,1] # false positives
# FN = confusion[1,0] # false negatives

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)


# In[144]:


# Let's plot accuracy sensitivity and specificity for various probabilities.
cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.show()


# From the curve above, 0.34 is the optimum point to take it as a cutoff probability.

# In[145]:


y_train_pred_final['final_predicted'] = y_train_pred_final.Converted_prob.map( lambda x: 1 if x > 0.34 else 0)

y_train_pred_final.head()


# ## 12. Assigning Lead Score to the Training data

# In[146]:


y_train_pred_final['Lead_Score'] = y_train_pred_final.Converted_prob.map( lambda x: round(x*100))

y_train_pred_final.head()


# -  **Model Evaluation**

# In[153]:


# Let's check the overall accuracy.
accuracy5=("Accuracy :",metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.final_predicted))
print(accuracy5)


# In[154]:


# Confusion matrix
confusion5 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_predicted )
confusion5


# In[155]:


TP = confusion5[1,1] # true positive 
TN = confusion5[0,0] # true negatives
FP = confusion5[0,1] # false positives 
FN = confusion5[1,0] # false negatives


# In[157]:


# Let's see the sensitivity of our logistic regression model
sensitivity=("Sensitivity : ",TP / float(TP+FN))
print(sensitivity)


# In[158]:


# Let us calculate specificity
print("Specificity :",TN / float(TN+FP))


# In[159]:


# Calculate false postive rate - predicting converted lead when the lead was actually not have converted
print("False Positive rate : ",FP/ float(TN+FP))


# In[160]:


# Positive predictive value 
print("Positive Predictive Value :",TP / float(TP+FP))


# In[161]:


# Negative predictive value
print("Negative Predictive Value : ",TN / float(TN+ FN))


# In[165]:


# Precision
TP / TP + FP

Precision5=("Precision : ",confusion5[1,1]/(confusion5[0,1]+confusion5[1,1]))
print(Precision5)


# In[166]:


# Recall
TP / TP + FN

recall5=("Recall :",confusion5[1,1]/(confusion5[1,0]+confusion5[1,1]))
print(recall5)


# In[167]:


p, r, thresholds = precision_recall_curve(y_train_pred_final.Converted, y_train_pred_final.Converted_prob)


# In[168]:


# plotting a trade-off curve between precision and recall
plt.plot(thresholds, p[:-1], "g-")
plt.plot(thresholds, r[:-1], "r-")
plt.show()


# **The above graph shows the trade-off between the Precision and Recall.**

# ## 13. Making predictions on the test set

# ### Scaling the test data

# In[169]:


X_test[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']] = scaler.transform(X_test[['TotalVisits',
                                                                                                        'Total Time Spent on Website',
                                                                                                        'Page Views Per Visit']])


# In[170]:


# Assigning the columns selected by the final model to the X_test 
X_test = X_test[col1]
X_test.head()


# In[171]:


# Adding a const
X_test_sm = sm.add_constant(X_test)

# Making predictions on the test set
y_test_pred = res.predict(X_test_sm)
y_test_pred[:10]


# In[172]:


# Converting y_test_pred to a dataframe which is an array
y_pred_1 = pd.DataFrame(y_test_pred)


# In[173]:


# Let's see the head
y_pred_1.head()


# In[174]:


# Converting y_test to dataframe
y_test_df = pd.DataFrame(y_test)


# In[175]:


# Putting Prospect ID to index
y_test_df['Prospect ID'] = y_test_df.index


# In[176]:


# Removing index for both dataframes to append them side by side 
y_pred_1.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)


# In[177]:


# Appending y_test_df and y_pred_1
y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)


# In[178]:


y_pred_final.head()


# In[179]:


# Renaming the column 
y_pred_final= y_pred_final.rename(columns={ 0 : 'Converted_prob'})


# In[180]:


# Rearranging the columns
y_pred_final = y_pred_final.reindex(columns=['Prospect ID','Converted','Converted_prob'])


# In[187]:


# Let's see the tail of y_pred_final
y_pred_final.tail(10)


# In[184]:


y_pred_final['final_predicted'] = y_pred_final.Converted_prob.map(lambda x: 1 if x > 0.34 else 0)


# In[186]:


y_pred_final.tail()


# In[188]:


# Let's check the overall accuracy.
print("Accuracy :",metrics.accuracy_score(y_pred_final.Converted, y_pred_final.final_predicted))


# In[189]:


# Making the confusion matrix
confusion2 = metrics.confusion_matrix(y_pred_final.Converted, y_pred_final.final_predicted )
confusion2


# In[190]:


TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# In[191]:


# Let's see the sensitivity of our logistic regression model
print("Sensitivity :",TP / float(TP+FN))


# In[192]:


# Let us calculate specificity
print("Specificity :",TN / float(TN+FP))


# In[195]:


y_pred_final['Lead_Score'] = y_pred_final.Converted_prob.map( lambda x: round(x*100))

y_pred_final.head()


# **Observations:**
# After running the model on the Test Data , we obtain:
# 
# -  **Accuracy : 83.6 %**
# -  **Sensitivity : 86.0 %**
# -  **Specificity : 82.1 %**

# ## 14. Results

# ## 1) Comparing the values obtained for Train & Test:

# **Train Data:**
# 
# -  Accuracy : 83.7 %
# -  Sensitivity : 84.8 %  
# -  Specificity : 82.9 %
# 
# **Test Data:**
# 
# -  Accuracy : 83.6 %
# -  Sensitivity : 86.0 %
# -  Specificity : 82.1 %

# <B> <FONT COLOR='GREEN'>Thus we have achieved our goal of getting a ballpark of the target lead conversion rate is more than 80% . The Model seems to predict the Conversion Rate very well and we should be able to give the CEO confidence in making good calls based on this model to get a higher lead conversion rate of 80%.</B> </FONT>

# ### 2) Finding out the leads which should be contacted:
# 

# **The customers which should be contacted are the customers whose "Lead Score" is equal to or greater than 85. They can be termed as 'Hot Leads'.**

# In[196]:


hot_leads=y_pred_final.loc[y_pred_final["Lead_Score"]>=85]
hot_leads


# <B> <FONT COLOR='GREEN'>So there are 573 leads which can be contacted and have a high chance of getting converted. The Prospect ID of the customers to be contacted are :</B> </FONT>

# In[197]:


print("The Prospect ID of the customers which should be contacted are :")

hot_leads_ids = hot_leads["Prospect ID"].values.reshape(-1)
hot_leads_ids


# ## 3) Finding out the Important Features from our final model:

# In[198]:


res.params.sort_values(ascending=False)


# ## 15. Recommendations

# -  <b> The company <FONT COLOR='GREEN'>should make calls</FONT> to the leads coming from the lead sources "Welingak Websites" and "Reference" as these are more likely to get converted.
# -  <b>The company <FONT COLOR='GREEN'>should make calls</FONT> to the leads who are the "working professionals" as they are more likely to get converted.
# -  <b>The company <FONT COLOR='GREEN'>should make calls</FONT> to the leads who spent "more time on the websites" as these are more likely to get converted.
# -  <b>The company <FONT COLOR='GREEN'>should make calls</FONT> to the leads coming from the lead sources "Olark Chat" as these are more likely to get converted.
# -  <b>The company <FONT COLOR='GREEN'>should make calls</FONT> to the leads whose last activity was SMS Sent as they are more likely to get converted.
# 
# -  <b>The company <FONT COLOR='RED'>should not make calls</FONT> to the leads whose last activity was "Olark Chat Conversation" as they are not likely to get converted.
# 
# -  <b>The company <FONT COLOR='RED'>should not make calls</FONT> to the leads whose lead origin is "Landing Page Submission" as they are not likely to get converted.
# -  <b>The company <FONT COLOR='RED'>should not make calls</FONT> to the leads whose Specialization was "Others" as they are not likely to get converted.
# -  <b>The company <FONT COLOR='RED'>should not make calls</FONT> to the leads who chose the option of "Do not Email" as "yes" as they are not likely to get converted.
