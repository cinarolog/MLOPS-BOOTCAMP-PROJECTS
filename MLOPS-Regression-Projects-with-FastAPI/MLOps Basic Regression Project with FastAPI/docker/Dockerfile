# Docker görüntüsü olarak Python'un 3.8.16-slim-bullseye sürümünü kullan
FROM python:3.8.16-slim-bullseye

# /code adında bir dizin oluştur
RUN mkdir /code

# Çalışma dizinini /code dizinine ayarla
WORKDIR /code

# requirements.txt dosyasını çalışma dizinine kopyala
COPY requirements.txt .

# requirements.txt dosyasındaki bağımlılıkları yükle
RUN pip install -r requirements.txt

# Tüm dosyaları çalışma dizinine kopyala
COPY . .

# Konteyner başlatıldığında çalışacak komutu belirle
CMD ["uvicorn", "main:app", "--host=0.0.0.0", "--port=8000"]
