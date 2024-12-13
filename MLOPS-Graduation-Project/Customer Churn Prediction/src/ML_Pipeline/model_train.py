from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
import pandas as pd



classifiers = {
                "SVM": SVC(),
                "Random Forest": RandomForestClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "KNN": KNeighborsClassifier(),
                "Logistic Regression": LogisticRegression(),
                "Naive Bayes": GaussianNB(),
                "Decision Trees": DecisionTreeClassifier(),
                "Neural Networks": MLPClassifier(),
                "AdaBoost": AdaBoostClassifier(),
                "XGBoost": XGBClassifier(),
                }

def train_all_algorithm_validate_with_val_data(x_train, x_val, y_train, y_val):
    global classifiers

    acc_list=[]
    precision_list=[]
    recall_list=[]
    f1score_list=[]
    name_list=[]
    cm_list=[]

    for name, model in classifiers.items():
        model.fit(x_train, y_train)
        y_val_pred = model.predict(x_val)
        
        accuracy = accuracy_score(y_val, y_val_pred)
        precision = precision_score(y_val, y_val_pred)
        recall = recall_score(y_val, y_val_pred)
        f1 = f1_score(y_val, y_val_pred)
        cm = confusion_matrix(y_val, y_val_pred)

        acc_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        f1score_list.append(f1)
        name_list.append(name)
        cm_list.append(cm)

        print(f"{name}: Accuracy = {accuracy:.2f}, Precision = {precision:.2f}, Recall = {recall:.2f}, F1 Score = {f1:.2f}")

    results= {
            "Model": name_list,
            "Accuracy": acc_list,
            "Precision": precision_list,
            "Recall": recall_list,
            "F1 Score": f1score_list
                }
    metric_df=pd.DataFrame(results)
    return metric_df



def last_model_train(model,X_train_scaled,x_test_scaled,Y_train,y_test):

    model=XGBClassifier()
    model.fit(X_train_scaled,Y_train)
    y_prediction=model.predict(x_test_scaled)

    accuracy = accuracy_score(y_test, y_prediction)
    precision = precision_score(y_test, y_prediction)
    recall = recall_score(y_test, y_prediction)
    f1 = f1_score(y_test, y_prediction)
    cm=confusion_matrix(y_test, y_prediction)

    score_df = pd.DataFrame({
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1}, index=["XGBoost"])

    return score_df,cm