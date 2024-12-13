import streamlit as st
import pickle
import numpy as np

# Modeli yükle
with open('best_xgb_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Streamlit uygulaması
def main():
    st.title("Customer Churn Prediction App")
    
    # Kullanıcı girişi al
    gender = st.radio("Cinsiyet Seçin:", ["Erkek", "Kadın"])
    age = st.number_input("Yaşınız:", min_value=0, max_value=100, value=30)
    no_of_days_subscribed = st.number_input("Abonelik Süresi (gün):", min_value=0, value=30)
    multi_screen = st.radio("Çoklu Ekran Kullanımı:", ["Evet", "Hayır"])
    mail_subscribed = st.radio("Mail Aboneliği:", ["Evet", "Hayır"])
    weekly_mins_watched = st.number_input("Haftalık İzleme Süresi (dakika):", min_value=0.0, value=60.0)
    minimum_daily_mins = st.number_input("Günlük Minimum İzleme Süresi (dakika):", min_value=0.0, value=30.0)
    weekly_max_night_mins = st.number_input("Haftalık Maksimum Gece İzleme Süresi (dakika):", min_value=0, value=120)
    videos_watched = st.number_input("İzlenen Video Sayısı:", min_value=0, value=10)
    customer_support_calls = st.number_input("Müşteri Destek Çağrısı Sayısı:", min_value=0, value=0)

    # Kullanıcının girdilerini modele uygun formatta düzenle
    gender = 1 if gender == "Erkek" else 0
    multi_screen = 1 if multi_screen == "Evet" else 0
    mail_subscribed = 1 if mail_subscribed == "Evet" else 0

    # Tahmin yap
    features = np.array([gender, age, no_of_days_subscribed, multi_screen, mail_subscribed,
                         weekly_mins_watched, minimum_daily_mins, weekly_max_night_mins,
                         videos_watched, customer_support_calls]).reshape(1, -1)
    prediction = model.predict(features)

    # Tahmini göster
    st.subheader("Tahmin Sonucu:")
    if prediction[0] == 1:
        st.write("Kullanıcı abonelikten ayrılabilir.")
    else:
        st.write("Kullanıcı abonelikten ayrılmayabilir.")

if __name__ == "__main__":
    main()
