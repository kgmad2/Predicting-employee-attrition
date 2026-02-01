# Assignment 1: Predicting Employee Attrition with Random Forests

## Overview

In this assignment, you'll build and compare decision tree and random forest models to predict employee attrition for a global consulting firm. You'll explore the dataset, train models, evaluate performance, and interpret feature importances to provide actionable insights for HR.

## Dataset

This assignment uses the [IBM HR Analytics Employee Attrition & Performance dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) from Kaggle, which contains employee information including demographics, job characteristics, satisfaction scores, and attrition status. The dataset is located in the `data/` folder.

## Assignment Goals

By completing this assignment, you will:
- Train and evaluate a baseline decision tree model
- Build a random forest ensemble model and compare performance
- Interpret feature importances to identify key drivers of employee attrition
- Communicate insights that HR leaders can act on

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- VS Code installed on your machine
- Git installed on your machine

### Step 1: Clone the Repository

```bash
git clone [YOUR_REPOSITORY_URL]
cd [YOUR_REPOSITORY_NAME]
```

### Step 2: Install Required Packages

Install the necessary Python packages by running:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

Or if using pip3:

```bash
pip3 install pandas numpy matplotlib seaborn scikit-learn
```

### Step 3: Open in VS Code

Open the project folder in VS Code:

```bash
code .
```

### Step 4: Run the Starter Code

Open `starter_notebook.py` in VS Code and run the file to verify your environment is set up correctly. You can run the file by:
- Right-clicking in the editor and selecting the play icon for the block of code of interest.
- Opening a terminal in VS Code (Terminal → New Terminal) and running:

```bash
python starter_notebook.py
```

Or:

```bash
python3 starter_notebook.py
```

You should see output confirming that libraries loaded successfully and the dataset displays correctly. You may have to select a runtime environment if you opt to run it using the play button.

## Repository Structure

```
├── data/
│   └── IBM_HR_Employee_Attrition.csv   # Dataset for the assignment
├── starter_notebook.py                 # Your main working file
├── README.md                           # This file
```

## Working on the Assignment

1. Open `starter_notebook.py` in VS Code
2. Follow the TODO comments to complete each step of the assignment
3. Run your code frequently to test your progress
4. Use the checkpoint messages to verify you're on track

## Submission

Once you've completed the assignment:

1. Save your work
2. Push to GitHub:
   ```bash
   git add .
   git commit -m 'completed employee attrition assignment'
   git push
   ```
3. Submit the link to your GitHub repository on the course platform

Your submission should include:
- A trained decision tree model and random forest model
- Model evaluation metrics (accuracy, precision, recall)
- A feature importance visualization
- A 150–200 word reflection on when random forests provide value over simpler models

## Getting Help

If you encounter issues:
- Check that all packages are installed correctly
- Verify the dataset file exists in the `data/` folder
- Review the assignment resources linked in the course
- Post questions in the course discussion forum

## Resources

The assignment references four resources that will guide you through:
1. How decision trees make predictions
2. What random forests are and why they work
3. Applying random forests in Python
4. Interpreting feature importance in random forests

These resources are available in your course materials.