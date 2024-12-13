from fastapi import FastAPI
import pickle
import pandas as pd
from pydantic import BaseModel

app = FastAPI()

class ModelSchema(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

@app.get("/")
def hello():
    return {"Hello": "Docker-MLOps"}

@app.post("/predict/knn")
def predict_knn(predict_value: ModelSchema):
    knn_filename = 'knn_model.sav'
    loaded_model = pickle.load(open(knn_filename, 'rb'))

    df = pd.DataFrame(
        [predict_value.dict()],
        columns=predict_value.dict().keys()
    )
    prediction = int(loaded_model.predict(df)[0])
    return {"prediction": prediction, "model": "KNN"}

@app.post("/predict/decision-tree")
def predict_decision_tree(predict_value: ModelSchema):
    decision_tree_filename = 'decision_tree_model.sav'
    loaded_model = pickle.load(open(decision_tree_filename, 'rb'))

    df = pd.DataFrame(
        [predict_value.dict()],
        columns=predict_value.dict().keys()
    )
    prediction = int(loaded_model.predict(df)[0])
    return {"prediction": prediction, "model": "Decision Tree"}

@app.post("/predict/logistic-regression")
def predict_logistic_regression(predict_value: ModelSchema):
    log_reg_filename = 'logistic_regression_model.sav'
    loaded_model = pickle.load(open(log_reg_filename, 'rb'))

    df = pd.DataFrame(
        [predict_value.dict()],
        columns=predict_value.dict().keys()
    )
    prediction = int(loaded_model.predict(df)[0])
    return {"prediction": prediction, "model": "Logistic Regression"}

# python -m uvicorn main:app --reload