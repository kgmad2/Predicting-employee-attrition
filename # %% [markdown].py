# %% [markdown]
# # Assignment 1: Predicting Employee Attrition with Random Forests
# 
# Execute cells **top to bottom**. Where you see `# TODO`, add your code.
# 
# **Data:** `data/IBM_HR_Employee_Attrition.csv`
# 
# **Deliverables produced in this file:**
# - Baseline Decision Tree metrics (accuracy, precision, recall)
# - Random Forest metrics + side-by-side comparison table
# - Feature importance visualization
# - Markdown sections for **Key Drivers of Attrition** and **Reflection**
# 
# ### Download Dependencies
# Run this cell once to install all dependencies. These can also be run directly in the terminal if you prefer.
# 

# %%
pip install pandas numpy matplotlib seaborn scikit-learn

# %%
print("Importing required libraries...")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

print("✓ All libraries imported successfully!\n")

# %% [markdown]
# ### Step 1: Load the dataset
# ----------------------------------------------------------------------------
# Confirm the CSV can be read and preview the first rows. The following code should output the first 5 rows of the IBM HR data.

# %%
# Set random seed for reproducibility
np.random.seed(42)

# Set plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

# Load the dataset
print("Loading employee attrition dataset...")
df = pd.read_csv('./data/IBM_HR_Employee_Attrition.csv')
print("✓ Dataset loaded successfully!\n")

# Display first few rows to verify load
print("First 5 rows of the dataset:")
print(df.head())

# Display basic info about the dataset
print("\nDataset Information:")
print(df.info())

print("\n" + "="*80)
print("CHECKPOINT: Verify that the dataset loaded correctly and you can see column names")
print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
print("="*80 + "\n")

# %% [markdown]
# ### Step 2: Explore and Prepare the Dataset
# ----------------------------------------------------------------------------
# Perform the same kind of exploratory analysis real data scientists do before building a model

# %%
df.describe()

# %%
df.info()

# %%
df['Attrition'].value_counts()

# %%
df_encoded = pd.get_dummies(df, drop_first=True)

print("Encoded dataset shape:", df_encoded.shape)

# %%
X_cleaned = df_encoded.drop('Attrition_Yes', axis=1)
y = df_encoded['Attrition_Yes']

# %%
print("\n" + "="*80)
print("CHECKPOINT: X_cleaned should have all numeric columns, y should contain Attrition values")
print(f"X_cleaned shape: {X_cleaned.shape if X_cleaned is not None else 'Not yet defined'}")
print(f"y shape: {y.shape if y is not None else 'Not yet defined'}")
print("="*80 + "\n")

# %% [markdown]
# ### Step 3: Train a Baseline Decision Tree Model
# ----------------------------------------------------------------------------
# Build a baseline decision tree for comparison

# %%
X_train, X_test, y_train, y_test = train_test_split(X_cleaned, y,test_size=0.2,random_state=42, stratify=y)
print( X_train.shape)
print(X_test.shape)

# %%
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_predictions = dt_model.predict(X_test)

# %%
dt_accuracy = accuracy_score(y_test, dt_predictions)
dt_precision = precision_score(y_test, dt_predictions)
dt_recall = recall_score(y_test, dt_predictions)
# Print results
print("\n" + "="*80)
print("BASELINE DECISION TREE RESULTS")
print("="*80)
# TODO: Print accuracy, precision, and recall with clear labels
print(f"Accuracy: {dt_accuracy if dt_accuracy is not None else 'Not yet calculated'}")
print(f"Precision: {dt_precision if dt_precision is not None else 'Not yet calculated'}")
print(f"Recall: {dt_recall if dt_recall is not None else 'Not yet calculated'}")
print("="*80 + "\n")

# %% [markdown]
# 

# %% [markdown]
# ### Step 4: Build and Evaluate a Random Forest Model
# ----------------------------------------------------------------------------
# Move beyond a single tree to a more powerful ensemble model

# %%
# Move beyond a single tree to a more powerful ensemble model

# Train a random forest classifier
# TODO: Initialize and train a RandomForestClassifier with these parameters:
# n_estimators=200, max_depth=None, min_samples_split=10, min_samples_leaf=2,
# max_features='sqrt', class_weight='balanced', random_state=42
rf_model = None  # Replace with trained RandomForestClassifier
rf_model = RandomForestClassifier( n_estimators=200, max_depth=None,min_samples_split=10,min_samples_leaf=2,max_features='sqrt',class_weight='balanced',random_state=42)
# Make predictions using probability threshold
# TODO: Use rf_model.predict_proba() to get probabilities for the positive class
# TODO: Apply a threshold of 0.35 to convert probabilities to predictions
# (rf_probabilities >= 0.35).astype(int)
rf_predictions = None  # Replace with threshold-adjusted predictions
rf_model.fit(X_train, y_train)

rf_probabilities = rf_model.predict_proba(X_test)[:, 1]

rf_predictions = (rf_probabilities >= 0.35).astype(int)

# Calculate evaluation metrics
# TODO: Calculate accuracy, precision, and recall for the random forest
rf_accuracy = None  # Replace with accuracy_score()
rf_precision = None  # Replace with precision_score()
rf_recall = None  # Replace with recall_score()

rf_accuracy = accuracy_score(y_test, rf_predictions)
rf_precision = precision_score(y_test, rf_predictions)
rf_recall = recall_score(y_test, rf_predictions)

# Print results
print("\n" + "="*80)
print("RANDOM FOREST RESULTS")
print("="*80)
print(f"Accuracy: {rf_accuracy if rf_accuracy is not None else 'Not yet calculated'}")
print(f"Precision: {rf_precision if rf_precision is not None else 'Not yet calculated'}")
print(f"Recall: {rf_recall if rf_recall is not None else 'Not yet calculated'}")
print("="*80 + "\n")

# Create comparison table
# TODO: Create a pandas DataFrame comparing both models side-by-side
# Columns: Model, Accuracy, Precision, Recall
model_comparison = pd.DataFrame({'Model': ['Decision Tree', 'Random Forest'],'Accuracy': [dt_accuracy, rf_accuracy],'Precision': [dt_precision, rf_precision],'Recall': [dt_recall, rf_recall]})

# TODO: Display the comparison table
print("\n" + "="*80)
print("MODEL COMPARISON")
print("="*80)
# Display table here
display(model_comparison)
print("="*80 + "\n")

# %% [markdown]
# ### Step 5: Interpret Feature Importances
# ----------------------------------------------------------------------------
# Turn model results into actionable insights for HR

# %%
# Extract feature importances
# TODO: Get feature_importances_ from rf_model and create a pandas Series
# with feature names as index
feature_importances = pd.Series(rf_model.feature_importances_,index=X_cleaned.columns)

# TODO: Sort feature importances in descending order
feature_importances=feature_importances.sort_values(ascending=False)

# TODO: Get top 10 most important features
top_10_features =  feature_importances.head(10)
print(top_10_features)
# Visualize top 10 feature importances
# TODO: Create a horizontal bar plot of the top 10 features
# Use plt.barh() or top_10_features.plot(kind='barh')
plt.figure()
top_10_features.sort_values().plot(kind='barh')
plt.title("Top 10 Feature Importances (Random Forest)")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.show()

print("\n" + "="*80)
print("KEY DRIVERS OF ATTRITION")
print("="*80)

# %% [markdown]
# #### Key Drivers of Attrition
# Write 3-5 bullet points explaining what the top factors reveal. Include at least one actionable takeaway for HR
# - [My first insight is that monthly income plays a role in employee attrition and in fact is a significant role amongst the others]
# - [Each feature is spread out amongst the importance scale. They are not all clumpes around the same importance levels]
# - [Most people don't mind driving a certain distant to work, as the features importance score is lower on the scale]
# - [I was surprised to see that hourly rate didn't have an impact on employee attrition especially with the knowledge of low wages]

# %% [markdown]
# ### Step 6: Reflection (150-200 words)
# ----------------------------------------------------------------------------
# Write a 150-200 word reflection addressing:
# - How the random forest improved upon the decision tree baseline
# - When ensemble methods are worth the added complexity
# - How these modeling skills connect to your final project

# %% [markdown]
# [The random forest allowed the data to be less skewed by allowing multiple options within the decision tree. With more options, comes more possibilities allowing the data to be more reliable and useful. Also, overfitting wil be less likely, because the system will take an average of more data instead of memorizing it.Ensemble methods are worth the added complexity, because the systems can handle complex data better and give a more accurate outcome. In my final project i will be integrating multiple modeling techniques to address a real-world business challenge. The techniques I am using here,which are building and evaluating ensemble models, interpreting variable importance, and communicating insights, will form the backbone of my final project later in the course.]

# %% [markdown]
# ### Step 7: Push to GitHub
# ----------------------------------------------------------------------------
# Once complete, save and push your work:
# 1. Save this file
# 2. Run in terminal:
# ```sh
# git add .
# git commit -m 'completed employee attrition assignment'
# git push
# ```


