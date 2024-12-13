####################################################### outlier #################################################
def extract_whiskers(data, whisker=1.5):
    median_value = np.median(data) # Medyan
    upper_quartile = np.percentile(data, 75) # 75%
    lower_quartile = np.percentile(data, 25) # 25% 

    iqr = upper_quartile - lower_quartile # Interquartile Range
    
    upper_whisker = data[data<=upper_quartile+whisker*iqr].max() # Maksimum Kabul Edilen Değer
    lower_whisker = data[data>=lower_quartile-whisker*iqr].min() # Minimum Kabul Edilen Değer
    
    print("Upper Whisker:", upper_whisker)
    print("Lower Whisker:", lower_whisker)


# local outlier factor

#df.loc



########################################### missing values #############################################

def mean_std_filling(column_name):    
    mean = column_name.mean()
    std = column_name.std()
    is_null = column_name.isna().sum()
    print('Mean:', round(mean,3), 'Std:', round(std,3), 'Null:', is_null)

    # Elimizdeki dizinin aritmetik ortalama ve standart sapma değerlerini kullanarak rastgele float veriler üretme
    rand_float = np.random.uniform(mean - std, mean + std, size = is_null)
    print('Numbers:', rand_float[:10])

    # Oluşturduğumuz sayılarla boş değerlerimizi doldurma
    column_name[np.isnan(column_name)] = rand_float
    column_name = column_name.astype(float)



# df.filna(nan,std,mean,)



################################################ numerical categorical####################################################3

# yıllara göre gruplayalım label encoding
datasets = [train, test]
for dataset in datasets:
    dataset.loc[dataset.Outlet_Establishment_Year >= 35, 'Outlet_Establishment_Year'] = 0
    dataset.loc[dataset.Outlet_Establishment_Year >= 23, 'Outlet_Establishment_Year'] = 1
    dataset.loc[dataset.Outlet_Establishment_Year >= 18, 'Outlet_Establishment_Year'] = 2
    dataset.loc[dataset.Outlet_Establishment_Year >= 13, 'Outlet_Establishment_Year'] = 3


catvars = df.select_dtypes(include=['object']).columns
numvars = df.select_dtypes(include = ['int32','int64','float32','float64']).columns

catvars,numvars


#################
df["mail_subscribed"]=df.mail_subscribed.map({"no":0,"yes":1})

#################
df['multi_screen'] = df['multi_screen'].apply(lambda x: str(x).replace('no', "0"))
df['multi_screen'] = df['multi_screen'].apply(lambda x: str(x).replace('yes', "1"))
df['multi_screen'] = df['multi_screen'].astype(int)

#################
for col in df.columns:
    df[col]=df[col].replace("Female",0)
    df[col]=df[col].replace("Male",1)


#################label encoding
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df["mail_subscribed"] = label_encoder.fit_transform(df["mail_subscribed"])


#################one hot encoding

data = {'mail_subscribed': ['no', 'yes', 'no', 'yes']}
df = pd.DataFrame(data)
df_encoded = pd.get_dummies(df, columns=['mail_subscribed'], prefix='mail_subscribed')


##############

def age_group(age):
    if age <= 6:
        return '1st'
    elif age <= 15:
        return '2nd'
    elif age <= 50:
        return '3rd'
    else:
        return '4th'

data = [train_df, test_df]
for dataset in data:
    dataset['Age Group'] = dataset.Age.apply(age_group)
    dataset.drop(columns='Age', inplace=True)        

############

def relatives(relative_number):
    if relative_number <= 3:
        return '1st'
    else:
        return '2nd'


data = [train_df, test_df]
for dataset in data:
    dataset['Relatives'] = dataset.Relatives.apply(relatives)


###############################################                     ###################################################















