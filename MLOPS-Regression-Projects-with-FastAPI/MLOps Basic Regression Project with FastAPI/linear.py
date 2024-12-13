
#%%
# Gerekli kütüphaneleri yükleyelim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pickle 

# Veri setini yükleyelim
data = pd.read_csv("advertising.csv")

# Veriyi inceleyelim
print(data.head())

# Bağımlı değişken (target) ve bağımsız değişkenleri (features) belirleyelim
X = data[['TV', 'Radio', 'Newspaper']]  # TV, radio ve newspaper harcamaları
y = data['Sales']  # Satış verileri

# Veriyi eğitim ve test setlerine bölelim
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Linear regresyon modelini oluşturalım
model = LinearRegression()

# Modeli eğitelim
model.fit(X_train, y_train)

# Test seti üzerinde tahmin yapalım
y_pred = model.predict(X_test)

# Model performansını değerlendirelim
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error (MSE): {mse}')
print(f'R-squared (R2): {r2}')

# Modelin katsayılarına ve interceptine bakalım
print('Katsayılar:', model.coef_)
print('Intercept:', model.intercept_)

# Tahminleri gerçek değerlerle karşılaştırmak için bir grafik çizelim
plt.scatter(y_test, y_pred)
plt.xlabel("Gerçek Satış")
plt.ylabel("Tahmin Satış")
plt.title("Gerçek vs Tahmin Satış Değerleri")
plt.show()


filename = 'model.pkl'

# Modeli kaydedin
with open(filename, 'wb') as file:
    pickle.dump(model, file)


# %%
