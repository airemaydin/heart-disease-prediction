# Heart Disease Risk Prediction - Machine Learning

## Overview
This repository contains a machine learning project developed to predict the risk of heart disease based on medical attributes. It demonstrates data science workflow, from data analysis and data preprocessing to model training and hyperparameter optimization. 

This project was built to showcase practical skills in applied machine learning and data handling using Python.

## Dataset
The dataset used for this project is the **Heart Failure Prediction Dataset** sourced from Kaggle.
* **Data Source:** [Kaggle Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)

## Technologies & Libraries
* **Language:** Python
* **Libraries:** Pandas (Data manipulation), Scikit-Learn (Machine learning models & preprocessing), Matplotlib & Seaborn (Data visualization).

## Key Steps & Methodology
1. **Exploratory Data Analysis (EDA):** Visualized feature distributions and their relationships with heart disease to understand underlying patterns.
2. **Data Preprocessing & Cleaning:** 
   * Handled invalid zero values in clinical measurements (e.g., RestingBP, Cholesterol) by replacing them with median values to preserve data integrity.
   * Encoded categorical variables into numerical formats using `LabelEncoder`.
3. **Preventing Data Leakage:** Implemented `train_test_split` *before* applying `MinMaxScaler`. This ensures the scaler only learns from the training data, preventing information from the test set from leaking into the model.
4. **Model Training & Optimization:** 
   * Trained a baseline K-Nearest Neighbors (KNN) classifier.
   * Applied `GridSearchCV` to search for the best hyperparameters (`n_neighbors` and `metric`), finalizing the most optimal and reliable model.
