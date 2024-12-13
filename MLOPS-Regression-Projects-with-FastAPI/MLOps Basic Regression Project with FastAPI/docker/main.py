from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle

# FastAPI uygulamasını oluştur
app = FastAPI()

# Pydantic BaseModel kullanarak model giriş verilerini tanımla
class ModelSchema(BaseModel):
    TV: float
    Radio: float
    Newspaper: float

# Ana sayfa için GET endpoint'i
@app.get("/")
def home():
    return {"message": "ML modele Hoşgeldiniz, predict için predict/linear kısmına gidiniz"}

# Lineer regresyon modeli kullanarak tahmin yapmak için POST endpoint'i
@app.post("/predict/linear")
def predict(predict_value: ModelSchema):
    # Eğitilmiş modelin dosya adı
    filename = "model.pkl"

    # Eğitilmiş modeli dosyadan yükle
    load_model = pickle.load(open(filename, "rb"))

    # Pydantic ModelSchema'dan alınan veriyi pandas DataFrame'e dönüştür
    df = pd.DataFrame([predict_value.dict()], columns=predict_value.dict().keys())

    # Tahmin yap
    prediction = load_model.predict(df)

    # Tahmini JSON formatında döndür
    return {"Prediction": float(prediction[0])}
