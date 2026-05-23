Heart Disease Prediction Using Machine Learning

Project Overview

This project predicts the possibility of heart disease using Machine Learning techniques. The project performs complete data analysis, visualization, model training, evaluation, and prediction using the Heart Disease dataset.

The project follows an end-to-end Data Science workflow including:

- Data Loading
- Data Cleaning
- Exploratory Data Analysis (EDA)
- Data Visualization
- Feature Selection
- Model Training
- Prediction
- Performance Evaluation

 Objectives

- Analyze patient health data
- Identify important factors affecting heart disease
- Build a predictive machine learning model
- Visualize insights using charts and graphs
- Predict heart disease risk for new patients

Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn

Project Structure

Heart-Disease-Prediction/

│── heart_prediction.py

│── heart.csv

│── requirements.txt

│── README.md

Dataset Information

Dataset: Heart Disease Dataset

Features included:

| Feature | Description |
|----------|-------------|
| age | Age of patient |
| sex | Gender |
| cp | Chest pain type |
| trestbps | Resting blood pressure |
| chol | Cholesterol level |
| fbs | Fasting blood sugar |
| restecg | Rest ECG result |
| thalach | Maximum heart rate |
| exang | Exercise induced angina |
| oldpeak | ST depression |
| slope | ST slope |
| ca | Major vessels |
| thal | Thalassemia |
| target | Heart disease presence |

Target:

- 0 → No Heart Disease
- 1 → Heart Disease

Project Workflow

Data Loading

Dataset loaded using Pandas:

```python
df = pd.read_csv("heart.csv")
