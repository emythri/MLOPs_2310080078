import numpy as np
import pandas as pd
import joblib

# Load the dataset
dataset = pd.read_csv("glass.csv")

# Splitting features and target
x = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]  # Use -1 to select the last column dynamically

# Train-test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

# Standardizing the features
from sklearn.preprocessing import StandardScaler  # Correct import

sc_x = StandardScaler()  # Instantiate the scaler
x_train = sc_x.fit_transform(x_train)  # Correct spelling: fit_transform
x_test = sc_x.transform(x_test)  # Transform test data using the same scaler

# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
cls = RandomForestClassifier(criterion='entropy', n_estimators=300, random_state=42)  # Correct spelling: entropy
cls.fit(x_train, y_train)

# Predictions and accuracy
y_pred = cls.predict(x_test)

print('ACCURACY is', cls.score(x_test, y_test) * 100, '%')

filename = 'final_model.sav'
joblib.dump(cls, filename)
