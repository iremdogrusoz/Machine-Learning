#pip install skompiler
import numpy as np
import pandas as pd
import seaborn as sns
import joblib
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


#pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)
#pd.set_option('display.float_format', lambda x: '%.3f' % x)
#pd.set_option('display.width', 500)

def load():
    data= pd.read_csv("diabetes.csv")
    return data
df=load()

# Exploratory data analysis

def check_df(dataframe, head=10, tail= 5):
    print("############################# SHAPE #############################")
    print(dataframe.shape)
    print("############################# TYPES #############################")
    print(dataframe.dtypes)
    print("############################# HEAD #############################")
    print(dataframe.head(head))
    print("############################# TAIL #############################")
    print(dataframe.tail(tail))
    print("############################# NA #############################")
    print(dataframe.isnull().sum())
    print("############################# QUANTILES #############################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99,1]). T)

check_df(df)

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

cat_cols, num_cols, cat_but_car= grab_col_names(df)

# Categorical and numerical variables analysis

def cat_summary(dataframe, col_name, plot= False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100* dataframe[col_name].value_counts()/len(dataframe)}))
    print("##################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data= dataframe)
        plt.show()

for col in cat_cols:
    cat_summary(df, col)

def num_summary(dataframe, numerical_col, plot= False):
    quantiles =[0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)
    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
    print(f"############### { col } ###############:")
    num_summary(df, col)

# Target Variable Analysis (The mean of numerical variables depending on target variable)

def num_target_summary(dataframe, target, numcols):
    print(dataframe.groupby(target).agg({numcols:'mean'}), end="\n\n\n")

for col in num_cols:
    num_target_summary(df, 'Outcome', col)

# Outliers
# Observation of outliers with a graph

sns.boxplot( x= df["Age"])
plt.show()

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

def has_outliers(dataframe, col_name):
    """
    Checks if the specified column in the dataframe has any outliers.

    Args:
        dataframe (pandas.DataFrame): The dataframe to check
        col_name (str): The name of the column to check.

    Returns:
        bool: True if the column has outliers, otherwise False
    """
    lower_limit, upper_limit = outliers_threshold(dataframe, col_name)
    return ((dataframe[col_name]> upper_limit) | (dataframe[col_name]< lower_limit)).any()

for col in num_cols:
    print( col, "----> ", has_outliers(df,col))

# Reaching Outliers

def get_outliers(dataframe, col_name, index=False):
    """
    Retrieves the outliers from the specified column in the dataframe

    Args:
        dataframe (pandas.DataFrame): The dataframe to retrieve outliers from
        col_name (str): The name of the column to retrieve outliers from.
        index (bool, optional): Whether to return the index of outliers. Defaults to False.

    Returns:
        pandas.DataFrame or pandas.Index: The dataframe or index containing the outliers
    """
    lower_limit, upper_limit = outliers_threshold(dataframe, col_name)
    outlier_df= dataframe[((dataframe[col_name] < lower_limit) | (dataframe[col_name] > upper_limit))]
    if outlier_df.shape[0] > 10:
        print(outlier_df.head())
    else:
        print(outlier_df)
    if  index:
        return outlier_df.index

for col in num_cols:
    print(col, get_outliers(df, col, True))

# Missing value analysis

def missing_values_table(dataframe, return_table=False):
    """
    Return columns that have missing values, along with the number of missing values and the percentage of missing values.

    Args:
        dataframe (pandas.DataFrame): Dataframe to analyze
        return_table (bool, optional): If True, return the missing value table. Otherwise, print the table. Defaults to False.

    Returns:
        pandas.DataFrame or None: If return_table is True, it returns missing value table. Otherwise, returns None. 
    """
    colums_with_missing_values= [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    total_misses =dataframe[colums_with_missing_values].isnull().sum()
    n_miss =total_misses.sort_values(ascending= False)
    percentage_missing = (total_misses / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(percentage_missing, 2)], axis=1, keys=['n_miss', 'percentage'])
    if return_table:
        return missing_df
    else:
        print(missing_df)

missing_values_table(df, True)

# Correlation Analysis
corr= df[num_cols].corr()
corr

sns.set(rc={"figure.figsize": (12, 12)})
sns.heatmap(corr, cmap="BuPu", annot=True) 
plt.show()

# Feature Engineering

# Procedures for missing and outlier values. 

# There is no missing values in the data but glucose, insulin, etc. contain 0. In the data, 0 might express missing values. Assingn 0 values as NAN

# Filling numerical variables with median
dff= df.apply(lambda x: x.fillna(x.mean()) if x.dtype!= "O" else x, axis=0)

## Filling categorical variables with mode
dff= df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique) <= 10) else x, axis= 0).isnull().sum()

# The relationship between missing values and dependent variable

na_cols= missing_values_table(df, True)

def missing_vs_target (dataframe, target, na_columns):
    temp_df= dataframe.copy()
    
    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(),1,0)
        
    na_flags= temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    
    for col in na_flags:
        group= temp_df.groupby(col)[target]
        print(pd.DataFrame({"TARGET_MEAN":group.mean(),"COUNT":group.count()}), end= "\n\n\n")
        
zero_to_nan_cols= ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in zero_to_nan_cols:
    df[col] = df[col].replace({'0':np.nan, 0:np.nan})

missing_vs_target(df, 'Outcome', zero_to_nan_cols)

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
        
for col in num_cols:
    replace_with_thresholds(df,col)

for col in num_cols:    
    print(col, has_outliers(df, col))

# Replacing NaN values with mean since the dataset contains only numerical variables except the target.
for col in zero_to_nan_cols:
    df[col].fillna(df[col].mean(), inplace= True)

# Local Outlier Factor
clf= LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df)

df_scores= clf.negative_outlier_factor_
df_scores[0:5]
# df_scores= -df_scores
sorted_df_scores= np.sort(df_scores)

scores= pd.DataFrame(np.sort(df_scores))
scores.plot(stacked= True, xlim= [0,50], style= '.-')
plt.show()

th= sorted_df_scores[7]

df.describe([0.01, 0.05, 0.75, 0.90, 0.99]).T
df[df_scores < th].index
df[df_scores< th].drop(axis=0, labels=df[df_scores < th].index)
# In order to delete above indexes
# df.drop(axis=0, labels=df[df_scores < th].index, inplace=True)

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

# Prediction for a New Observation
random_user = X.sample(1, random_state=45)
voting_clf.predict(random_user)

joblib.dump(voting_clf, "voting_clf2.pkl")

new_model = joblib.load("voting_clf2.pkl")
new_model.predict(random_user)
