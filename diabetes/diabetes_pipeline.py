import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# Categorical and numerical variables
def grab_col_names(dataframe, cat_th=10, car_th=20 ):
    # Find categorical columns
    cat_cols= [col for col in dataframe.columns if dataframe[col].dtypes == 'O']
    
    #Find numerical columns with low cardinality (i.e., number of unique values less than categorical threshold)
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th
                   and dataframe[col].dtypes != 'O']
    
    # Find categorical columns with high cardinality (i.e., number of unique values greater than cardinality threshold)
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique()> car_th
                   and dataframe[col].dtypes == 'O']
    
    # Combine categorical and numerical columns with low cardinality(exclude categorical columns with high cardinality)
    cat_cols= cat_cols + num_but_cat
    cat_cols= [col for col in cat_cols if col not in cat_but_car]
    
    # Find numerical columns (exclude numerical columns with low cardinality)
    num_cols= [col for col in dataframe.columns if dataframe[col].dtypes != 'O']
    num_cols= [col for col in num_cols if col not in num_but_cat]
    
    # Print Summary
    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")
    return cat_cols, num_cols, cat_but_car

# Checking Outliers
def outliers_threshold (dataframe, col_name, q1= 0.25, q3= 0.75):
    """
    Calculates the upper and lower limits for identifying outliers in a given column of a Pandas dataframe.
    Args:
        dataframe (pandas.DataFrame): The dataframe to calculate outliers for.
        col_name (str): The name of the column to calculate outliers for.
        q1 (float, optional): The quantile to use for the first quartile. Defaults to 0.25.
        q3 (float, optional): The quantile to use for the third quartile. Defaults to 0.75.

    Returns:
        tuple: A tuple of the lower and upper outlier thresholds for the specified column.
    """
    q1_val= dataframe[col_name].quantile(q1)
    q3_val= dataframe[col_name].quantile(q3)
    iqr= q3_val - q1_val
    upper_limit= q3_val + 1.5* iqr
    lower_limit = q3_val - 1.5 * iqr
    return lower_limit, upper_limit

# Replacing outliers with thresholds
def replace_with_thresholds(dataframe, column_name):
    lower, upper= outliers_threshold(dataframe, column_name)
    
    replaced= False
    if dataframe[column_name].max() > upper:
        dataframe.loc[dataframe[column_name]> upper, column_name]= upper
        replaced= True
    if dataframe[column_name].min() < lower:
        dataframe.loc[dataframe[column_name] < lower, column_name] = lower
        replaced= True
        
    return replaced

# One-hot Encoding
def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe= pd.get_dummies(dataframe, columns= categorical_cols, drop_first=drop_first)
    return dataframe


def diabetes_data_prep(dataframe):
    dataframe.columns= [col.upper() for col in dataframe.columns]
    
    #Glucose
    dataframe['NEW_CAT_GLUCOSE'] = pd.cut(x=dataframe['GLUCOSE'], bins= [-1,99,125, 400], labels= ['normal', 'prediabetes', 'diabetes'])
    
    #Age
    df_age= dataframe['AGE']
    new_age= "NEW_CAT_AGE"
    dataframe.loc[(df_age <39),new_age ] = 'young adults'
    dataframe.loc[(df_age >= 39) & df_age < 60, new_age] = 'middle-aged adults'
    dataframe.loc[(df_age>=60), new_age]= 'old adults'
    
    #BMI
    dataframe['NEW_BMI_RANGE']= pd.cut(x=dataframe['BMI'], bins= [-1, 18.5, 24.9, 29.9, 100], labels= ['underweight', 'healthy', 'overweight', 'obese'])
    
    # Blood Pressure
    dataframe['NEW_BLOOD_PRESSURE']= pd.cut(x=dataframe['BLOODPRESSURE'], bins= [-1, 60, 80, 89, 120], labels= ['low', 'normal', 'hypertension1', 'hypertension2'])
    
    #Refreshing cat_cols
    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe, cat_th=5, car_th=20)
    cat_cols= [col for col in cat_cols if "OUTCOME" not in col]
    
    # One-hot encoding
    df= one_hot_encoder(dataframe, cat_cols, drop_first= True)
    
    # Standardize numerical variables.
    X_scaled = StandardScaler().fit_transform(df[num_cols])
    df[num_cols] = pd.DataFrame( X_scaled, columns= df[num_cols].columns)
    
    y= df["OUTCOME"]
    X= df.drop(["OUTCOME"], axis=1)
    
    return X,y

# Base Models
def base_models (X,y,scoring="roc_auc"):
    print("Base Models...")
    classifiers= [('LR', LogisticRegression()),
                 ('KNN', KNeighborsClassifier()),
                 ('SVC', SVC()),
                 ('CART', DecisionTreeClassifier()),
                 ('RF', RandomForestClassifier()),
                 ('Adaboost', AdaBoostClassifier()),
                 ('GBM', GradientBoostingClassifier()),
                 ('XGboost', XGBClassifier(eval_metric='logloss')),
                 ('LightGBM', LGBMClassifier())]
    for name, classifier in classifiers:
        cv_results= cross_validate(classifier, X, y, cv=3, scoring=scoring)
        print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name}) ")
        
# Hyperparameter Optimization
knn_params = {"n_neighbors": range(2,60)}
cart_params={"max_depth": range(1,20),
             "min_samples_split": range(2,30)}

rf_params= {"max_depth": [8,15,None],
            "max_features": [5,7,"sqrt"],
            "min_samples_split": [15,20],
            "n_estimators": [100,200]}
xgboost_params= {"learning_rate": [0.1,0.01,0.5],
                 "max_depth":[5,8],
                 "n_estimators": [100,200]}
lightgbm_params= {"learning_rate": [0.01, 0.1],
                  "n_estimators": [300, 500, 1000]}
classifiers= [( 'KNN', KNeighborsClassifier(), knn_params), 
               ('CART', DecisionTreeClassifier(), cart_params),
               ('RF', RandomForestClassifier(), rf_params),
               ('XGboost', XGBClassifier(eval_metric='logloss'), xgboost_params),
               ('LightGBM', LGBMClassifier(), lightgbm_params)
               ]

def hyperparameter_optimization(X,y,cv=3, scoring="roc_auc"):
    print("Hyperparameter Optimization...")
    best_models = {}
    for name, classifier, params in classifiers:
        print(f"########## {name} ##########")
        cv_results= cross_validate(classifier,X,y, cv=cv, scoring= scoring)
        print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")
        
        gs_best= GridSearchCV(classifier, params,cv=cv, n_jobs= -1, verbose= False).fit(X,y)
        final_model= classifier.set_params(**gs_best.best_params_)
        
        cv_results= cross_validate(final_model, X,y, cv=cv, scoring=scoring)
        print(f"{scoring} (After): {round(cv_results['test_score'].mean(),4)}")
        best_models[name]= final_model
    return best_models

# Stacking and Ensemble Learning
def voting_classifier(best_models, X,y):
    """
    Trains a voting classifier that combines the predictions of multiple models and returns the trained model object.

    Args:
        best_models (dict): A dictionary containing the best trained models for each model type. The keys are model type names (e.g., "KNN", "RF", "LightGBM") and the values are the actual model objects.
        X (array-like of shape): The input data.
        y (array-like of shape): The target values.
        
        Returns
        -------
        voting_clf (VotingClassifier): The trained voting classifier object.
    """
    
    print("Voting Classifier...")
    
    # Create a VotingClassifier object that combines the predictions of the best models using soft voting.
    #The estimators argument is a list of tuples, where each tuple contains a string identifier for the model and the actual model object.
    # Train the voting classifier on the input data using the fit() method.
    voting_clf= VotingClassifier(estimators= [('KNN', best_models["KNN"]),
                                              ('RF', best_models["RF"]),
                                              ('LightGBM', best_models["LightGBM"])], voting='soft').fit(X,y)
    # Evaluate the performance of the voting classifier using cross-validation.
    # The cross_validate() function returns a dictionary with the test scores for each metric and each fold.
    cv_results= cross_validate(voting_clf, X,y, cv=3, scoring=["accuracy", "f1", "roc_auc"])
    
    # Compute the mean test scores across all folds and print them.
    
    print(f"Accuracy: {cv_results['test_accuracy'].mean()}")
    print(f"F1Score: {cv_results['test_f1'].mean()}")
    print(f"ROC_AUC: {cv_results['test_roc_auc'].mean()}")
    
    # Return the trained voting classifier object.
    return voting_clf

# Pipeline Main Function
def main():
    # Load the dataset and prepare the data.
    df= pd.read_csv("diabetes.csv")
    X, y = diabetes_data_prep(df)
    base_models(X,y)
    
    # Perform hyperparameter optimization.
    best_models= hyperparameter_optimization(X,y)
    
    # Train a voting classifier using the best models.
    voting_clf= voting_classifier(best_models, X,y)
    
    # Save the trained voting classifier to a file
    joblib.dump(voting_clf, "voting_clf.pkl")
    
    # Return the trained voting classifier object.
    return voting_clf

if __name__ == "__main__":
    print("Process is started")
    main()
    print("Done!")
