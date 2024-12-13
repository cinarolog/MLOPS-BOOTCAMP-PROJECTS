from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

app = FastAPI()

class ModelSchema(BaseModel):
    gender: int
    age: int
    no_of_days_subscribed: int
    multi_screen: int
    mail_subscribed: int
    weekly_mins_watched: float
    minimum_daily_mins: float
    weekly_max_night_mins: int
    videos_watched: int
    customer_support_calls: int
   

@app.get("/")
def hello():
    return {"Merhaba": "MLOps-Docker Projesine Hoşgeldiniz."}

@app.post("/predict/knn")
def predict_churn(predict_value: ModelSchema):
    knn_filename = 'knn_model.pkl'
    loaded_model = loaded_model = joblib.load(knn_filename)

    df = pd.DataFrame(
        [predict_value.dict()],
        columns=predict_value.dict().keys()
    )
    prediction = int(loaded_model.predict(df)[0])
    return {"prediction": prediction, "model": "KNN"}


# {
#   "gender": 0,
#   "age": 36,
#   "no_of_days_subscribed": 62,
#   "multi_screen": 0,
#   "mail_subscribed": 0,
#   "weekly_mins_watched": 148.35,
#   "minimum_daily_mins": 12.2,
#   "weekly_max_night_mins": 82,
#   "videos_watched": 1,
#   "customer_support_calls": 1
# }

# python -m uvicorn main:app --reload
# docker login
# docker tag mlops-app dataloper/mlops-app
# docker push dataloper/mlops-app

