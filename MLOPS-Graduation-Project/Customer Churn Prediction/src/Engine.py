import matplotlib.pyplot as plt
import seaborn as sns
from ML_Pipeline.utils import read_data, inspection, null_values, drop_column
from ML_Pipeline.visualization import correlation, visualization_performances
from ML_Pipeline.model_train import train_all_algorithm_validate_with_val_data, classifiers, last_model_train
from ML_Pipeline.split_data import split_data_test, split_data_val
from ML_Pipeline.standardization import scaling
from ML_Pipeline.hyperparameters import hyper_parameters
from ML_Pipeline.cross_validation import cross_validation
from ML_Pipeline.encoding import split_cat_num, get_unique, encoding_column, independent_dependent

# İlk olarak, gerekli kütüphaneleri içeri aktarıyoruz.
# Bu kütüphaneler veri işleme, model eğitimi ve değerlendirmesi için kullanılacak.
# Ayrıca, özel modüller ve fonksiyonlar da içeri aktarılıyor.

# Ardından, veri setini okuyoruz ve bazı ön işlemler yapıyoruz:
df = read_data("../data/data_regression.csv")
df = null_values(df)  # Null değerleri temizliyoruz.
inspection(df)  # Veriyi inceliyoruz ve gerekirse temizleme işlemleri yapıyoruz.

# Bazı sütunları dropluyoruz çünkü eğitimde kullanmak için uygun değiller:
dropped_columns = ["year", "customer_id", "phone_no"]
drop_column(df, dropped_columns)

# Korelasyon matrisini çizip yüksek korelasyonlu sütunları dropluyoruz:
correlation(df)
high_corr = ["maximum_days_inactive", "maximum_daily_mins"]
drop_column(df, high_corr)

# Kategorik ve sayısal sütunları ayırıyoruz:
numeric_columns, categoric_columns = split_cat_num(df)

# Kategorik sütunlardaki benzersiz değerlere bakıyoruz ve bunları kodluyoruz:
get_unique(df, categoric_columns)
encoding_column(df, "gender")
encoding_column(df, "multi_screen")
encoding_column(df, "mail_subscribed")

# Bağımsız ve bağımlı değişkenleri belirliyoruz:
X, y = independent_dependent(df)

# Veriyi eğitim ve test setlerine ayırıyoruz:
X_train, x_test, Y_train, y_test = split_data_test(X, y)

# Eğitim setini daha sonra kullanmak üzere eğitim ve doğrulama setlerine ayırıyoruz:
x_train, x_val, y_train, y_val = split_data_val(X_train, Y_train)

# Tüm algoritmaları eğitip doğrulama seti üzerinde değerlendiriyoruz:
metric_df = train_all_algorithm_validate_with_val_data(x_train, x_val, y_train, y_val)

# Model performanslarını görselleştiriyoruz:
visualization_performances(metric_df)

# Veriyi standartlaştırıyoruz:
x_train_scaled, x_val_scaled = scaling(x_train, x_val)

# Standartlaştırılmış veri ile tüm modelleri eğitip doğrulama seti üzerinde değerlendiriyoruz:
metric_df2 = train_all_algorithm_validate_with_val_data(x_train_scaled, x_val_scaled, y_train, y_val)

# Standartlaştırılmış model performanslarını görselleştiriyoruz:
visualization_performances(metric_df2)

# XGBoost modeli üzerinde çapraz doğrulama yapıyoruz:
xgb = classifiers["XGBoost"]
cross_validation(xgb, X, y)

# Veriyi standartlaştırıp XGBoost modelini eğitip test seti üzerinde değerlendiriyoruz:
X_train_scaled, x_test_scaled = scaling(X_train, x_test)
xgb_score_df, cm = last_model_train(xgb, X_train_scaled, x_test_scaled, Y_train, y_test)

# Hyperparameter optimizasyonu yaparak en iyi modeli seçiyoruz:
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}
best_parameters, best_score = hyper_parameters(xgb, param_grid, X_train, Y_train)
best_xgb_model = best_parameters

# En iyi parametrelerle XGBoost modelini tekrar eğitip test seti üzerinde değerlendiriyoruz:
best_xgb_model_score_df, cm = last_model_train(best_xgb_model, X_train_scaled, x_test_scaled, Y_train, y_test)

# Sonuçları görselleştiriyoruz:
sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', xticklabels=['Not Churn', 'Churn'],
            yticklabels=['0', '1'])
plt.title('Confusion Matrix')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.show()
