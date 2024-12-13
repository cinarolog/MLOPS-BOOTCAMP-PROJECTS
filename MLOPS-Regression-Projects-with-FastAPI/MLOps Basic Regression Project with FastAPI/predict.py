import pickle

# Eğitilmiş modeli dosyadan yükle
model = pickle.load(open("model.pkl", "rb"))

# Tahmin yapmak için kullanılacak örnek veri
example_data = [[230.5, 38.4, 78.3]]

# Model üzerinde tahmin yap
prediction = model.predict(example_data)

# Tahmin sonucunu ekrana yazdır
print("Prediction:", prediction)
