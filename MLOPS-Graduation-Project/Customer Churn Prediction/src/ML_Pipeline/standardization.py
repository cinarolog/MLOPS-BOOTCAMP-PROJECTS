from sklearn.preprocessing import StandardScaler

def scaling(x_train,x_val):
    # StandardScaler nesnesini oluştur
    scaler = StandardScaler()
    # Eğitim verilerine fit ve transform uygula
    x_train_scaled = scaler.fit_transform(x_train)
    # Doğrulama (validation) verilerine transform uygula (fit etme, sadece eğitim verilerine yapılır)
    x_val_scaled = scaler.transform(x_val)
    return x_train_scaled,x_val_scaled