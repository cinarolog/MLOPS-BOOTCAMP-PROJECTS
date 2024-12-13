import pandas as pd

# Veri dosyasını okumak için bir fonksiyon
def read_data(file_path, **kwargs):
    # Pandas kütüphanesini kullanarak CSV dosyasını oku
    raw_data = pd.read_csv(file_path, **kwargs)
    return raw_data

# Veri çerçevesini incelemek için bir fonksiyon
def inspection(dataframe):
    # Değişken türlerini göster
    print("Çalıştığımız değişken türleri:")
    print(dataframe.dtypes)

    # Veri çerçevesinin genel bilgilerini göster
    print("\nVeri çerçevesinin genel bilgileri:")
    print(dataframe.info())

    # Veri çerçevesinin ilk birkaç satırını göster
    print("\nVeri çerçevesinin ilk birkaç satırı:")
    print(dataframe.head())

    # Veri çerçevesinin boyutunu (satır ve sütun sayısı) göster
    print("\nVeri çerçevesinin boyutu:")
    print(dataframe.shape)

    # Temel istatistiksel bilgileri (count, mean, std, min, 25%, 50%, 75%, max) göster
    print("\nTemel istatistiksel bilgiler:")
    print(dataframe.describe().T)

    # Eksik değerlere sahip toplam örnek sayısını göster
    print("\nEksik değerlere sahip toplam örnek sayısı:")
    print(dataframe.isnull().any(axis=1).sum())

    # Her değişkendeki toplam eksik değer sayısını göster
    print("\nHer değişkendeki toplam eksik değer sayısı:")
    print(dataframe.isnull().sum())


# Null değerleri kaldırmak için bir fonksiyon
def null_values(df):
    if df is None:
        print("Hata: DataFrame NoneType olarak tanımlandı.")
        return None
    # Eksik değerlere sahip satırları kaldır
    df = df.dropna()
    return df


def drop_column(df,df_column_list):
    """
        df_column_list = droplanacak colonları liste içinde veriniz.
        örnek >>>>>   df_column_list=["a","b"]  ya da bir taneyse df_column_list=["a"]
    """
    df.drop(df_column_list, axis=1, inplace=True)
    df.info()
    return df