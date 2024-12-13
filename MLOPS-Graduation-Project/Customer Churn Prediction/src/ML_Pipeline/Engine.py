# Gerekli kütüphaneleri içe aktar
import pickle
from ML_Pipeline.utils import read_data, inspection, null_values,drop_column
from ML_Pipeline.visualization import correlation,visualization_performances
from ML_Pipeline.model_train import train_all_algorithm_validate_with_val_data,classifiers,last_model_train
from ML_Pipeline.split_data import split_data_test,split_data_val
from ML_Pipeline.standardization import scaling
from ML_Pipeline.hyperparameters import hyper_parameters
from ML_Pipeline.cross_validation import cross_validation
from ML_Pipeline.encoding import split_cat_num,get_unique,encoding_column,independent_dependent
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV

# İlk veri setlerini oku
df = read_data("../data/data_regression.csv")

# Null değerleri düşürme
df = null_values(df)

# Veriyi inceleme ve temizleme
inspection(df)

#customer_id ,year ve phone_no  modelm eğitimimizde yeterli bilgiyi sağlamayacağından dolayı droplamak güzel bir çözüm.
dropped_columns = ["year","customer_id","phone_no"]
drop_column(df,dropped_columns)

# correlation matrix
correlation(df)

#drop high correlated columns
high_corr=["maximum_days_inactive","maximum_daily_mins"]
drop_column(df,high_corr)

# split cateoric column and numeric column
numeric_columns,categoric_columns=split_cat_num(df)

# Categoric columns unique values
get_unique(df,categoric_columns)

#encodig categoric columns
encoding_column(df,"gender")
encoding_column(df,"multi_screen")
encoding_column(df,"mail_subscribed")

# İndependen-dependent variables X,y
X,y=independent_dependent(df)

# Train-test split 80-20
X_train, x_test, Y_train, y_test=split_data_test(X,y)

# train-val split 60-20
x_train, x_val, y_train, y_val=split_data_val(X_train,Y_train)

#train all model validate with validation data
metric_df=train_all_algorithm_validate_with_val_data(x_train, x_val, y_train, y_val)

#visualize model performance
visualization_performances(metric_df)

# standardization-scaling
x_train_scaled,x_val_scaled=scaling(x_train,x_val)

#train all model validate with standardizate data
metric_df2=train_all_algorithm_validate_with_val_data(x_train_scaled, x_val_scaled, y_train, y_val)

#visualize model performance
visualization_performances(metric_df2)


#cross validation xgboost
"""
    Görselleştirmelerden sonra gördük ki en iyi algoritmamız XGBOOST.
    Birde cross validation deneyelim.Bakalım gerçektende XGBOOST datanın tümü için
    başarılı olabilecekmi.
"""

#modelimiz için instance oluşturalım
xgb=classifiers["XGBoost"]

cross_validation(xgb,X,y)

# standardization-scaling
X_train_scaled,x_test_scaled=scaling(X_train,x_test)

# Train=train+val test=test  80-20
xgb_score_df,cm=last_model_train(xgb,X_train_scaled,x_test_scaled,Y_train,y_test)

#Hyper parameteres Optimization

param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }

best_parameters,best_score=hyper_parameters(xgb,param_grid,X_train,Y_train)

# Get the best model from the grid search
best_xgb_model = best_parameters

# Again Train with best parameters
best_xgb_model_score_df,cm=last_model_train(best_xgb_model,X_train_scaled,x_test_scaled,Y_train,y_test)

sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', xticklabels=['Not Churn', 'Churn'], yticklabels=['0', '1'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
