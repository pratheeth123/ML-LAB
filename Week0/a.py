import pandas as pd
# Load dataset
df = pd.read_csv("loan_data.csv")
# Preprocessing
df.fillna(df.mean(numeric_only=True), inplace=True)
Machine Learning Workflow: From Raw Data to Model Evaluation 17
df = pd.get_dummies(df, drop_first=True)
X = df.drop("Default", axis=1)
y = df["Default"]
from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(X)
# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, tes
t_size=0.2)
# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
lr = LogisticRegression().fit(X_train, y_train)
rf = RandomForestClassifier().fit(X_train, y_train)
# Predictions
p1 = lr.predict(X_test)
p2 = rf.predict(X_test)
# Evaluation
from sklearn.metrics import f1_score
print("Logistic Regression F1:", f1_score(y_test, p1))
print("Random Forest F1:", f1_score(y_test, p2))
