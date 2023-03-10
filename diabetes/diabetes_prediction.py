import joblib
import pandas as pd

# Load the dataset
df = pd.read_csv("datasets/diabetes.csv")

# Choose a random patient from the dataset
random_user = df.sample(1, random_state=45)

# Load the trained model
new_model = joblib.load("voting_clf.pkl")

# Make a prediction for the chosen patient.
new_model.predict(random_user)

from diabetes_pipeline import diabetes_data_prep
# Prepare the data before prediction
X, y = diabetes_data_prep(df)
random_user = X.sample(1, random_state=50)
new_model = joblib.load("voting_clf.pkl")
new_model.predict(random_user)
