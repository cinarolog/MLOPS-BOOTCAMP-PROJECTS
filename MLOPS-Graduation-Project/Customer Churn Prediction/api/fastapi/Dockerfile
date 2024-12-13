# Python 3.8 tabanlı bir görüntü kullan
FROM python:3.8

# Çalışma dizinini oluştur ve ayarla
RUN mkdir /code
WORKDIR /code

# Yereldeki requirements.txt dosyasını kopyala
COPY requirements.txt .

# Bağımlılıkları yükle
RUN pip install -r requirements.txt

# Tüm dosyaları kopyala
COPY . .

# Uygulamayı başlat
CMD ["uvicorn", "main:app", "--host=0.0.0.0", "--port=8080"]


# docker build -t mlops-app .
# docker run -p 8080:8080 mlops-app

# docker run -p 8080:8080 dataloper/mlops-app
