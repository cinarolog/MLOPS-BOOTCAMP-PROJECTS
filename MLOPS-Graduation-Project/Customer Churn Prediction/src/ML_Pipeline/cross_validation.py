from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

def cross_validation(model,X,y):
    # StandardScaler nesnesini oluştur
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Cross-validation için StratifiedKFold kullanımı
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scorings=["accuracy","precision","recall","f1"]

    for scoring in scorings:

        print("---------------"+scoring+"---------------")
        # Cross-validation işlemi
        scores = cross_val_score(model, X_scaled, y, cv=cv, scoring=scoring)
        # Cross-validation sonuçlarını ekrana yazdırma
        print("Cross-Validation Scores:", scores)
        print("Ortalama Score:", scores.mean())
