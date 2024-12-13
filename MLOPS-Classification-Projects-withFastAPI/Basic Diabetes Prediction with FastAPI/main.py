from fastapi import FastAPI, Path
import pickle
import pandas as pd
from pydantic import BaseModel

app = FastAPI()

class modelSchema(BaseModel):
    Pregnancies:int
    Glucose:int
    BloodPressure:int
    SkinThickness:int
    Insulin:int
    BMI:float
    DiabetesPedigreeFunction:float
    Age:int

@app.get("/")
def hello():
    return {"Hello": "Docker-MLOps"}

@app.post("/predict/knn")
def create_student(predict_value: modelSchema):
    filename = 'knn_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))

    df = pd.DataFrame(
        [predict_value.dict().values()],
        columns=predict_value.dict().keys()
        )
    pred=loaded_model.predict(df)
    return {"predict":int(pred[0])}

# python -m uvicorn main:app --reload
