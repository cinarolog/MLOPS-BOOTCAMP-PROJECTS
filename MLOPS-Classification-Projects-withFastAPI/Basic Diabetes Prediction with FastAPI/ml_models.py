#%%
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle

# Load the dataset
df = pd.read_csv("diabetes.csv", encoding="utf-8")

# Separate features and target variable
y = df['Outcome']
X = df.drop(['Outcome'], axis=1)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# K-Nearest Neighbors (KNN) model
knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(x_train, y_train)
print("KNN Score:", knn.score(x_test, y_test))

# Decision Tree model
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(x_train, y_train)
print("Decision Tree Score:", decision_tree.score(x_test, y_test))

# Logistic Regression model
log_reg = LogisticRegression(random_state=42)
log_reg.fit(x_train, y_train)
print("Logistic Regression Score:", log_reg.score(x_test, y_test))

# Save models to files
knn_filename = 'knn_model.sav'
decision_tree_filename = 'decision_tree_model.sav'
log_reg_filename = 'logistic_regression_model.sav'

pickle.dump(knn, open(knn_filename, 'wb'))
pickle.dump(decision_tree, open(decision_tree_filename, 'wb'))
pickle.dump(log_reg, open(log_reg_filename, 'wb'))

print("Models trained and saved successfully!")

# %%
