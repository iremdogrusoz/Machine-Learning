# Diabetes Prediction Model
This is a machine learning project that trains a voting classifier to predict whether a person has diabetes based on several input features such as age, BMI and blood pressure.

## Dataset
The diabetes dataset used in this project is the Pima Indians Diabetes Database, which is publicly available from the UCI Machine Learning Repository. The dataset contains information about 768 female patients of Pima Indian heritage, including their age, BMI, blood pressure, and other health metrics, as well as a binary indicator of whether they have diabetes. The dataset was originally collected by the National Institute of Diabetes and Digestive and Kidney Diseases and has been used extensively in machine learning research. 
The dataset can be found at the following URL: https://www.kaggle.com/uciml/pima-indians-diabetes-database

## Getting Started
To use this project, you'll need to have Python 3 installed on your computer, along with the following Python packages:
- numpy
- pandas
- seaborn
- joblib
- matplotlib
- scikit-learn
- lightgbm
- xgboost


## Project Structure
The project has the following files:
- 'diabetes.csv': The dataset used in the project.
- 'diabetes_research': The research script for data exploration and preprocessing.
- 'diabetes_pipeline': The main Python script that trains the model.
- 'diabetes_prediction': The prediction script uses loaded model file for prediction.

## Results
The trained voting classifier achieved an roc auc score of 0.84 on the test set, which indicates that it is moderately successful in predicting whether a person has diabetes.

## Conclusion
This project demonstrates the use of scikit-learn, lightgbm, and xgboost to train a machine learning model on the Pima Indians Diabetes Database. By preprocessing the data, training several base models, performing hyperparameter optimization, and boosting the performance of the model with a voting classifier, we were able to achieve moderate success in predicting.
