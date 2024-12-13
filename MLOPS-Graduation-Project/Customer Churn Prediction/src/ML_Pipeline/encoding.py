import pandas as pd

def split_cat_num(df):
    numeric_columns = df.select_dtypes(include='number')
    categoric_columns = df.select_dtypes(include='object')
    return numeric_columns,categoric_columns



def get_unique(df,categoric_columns):

    for col in categoric_columns.columns:
        print("----------"+col+"-----------")
        print(df[col].unique())


def encoding_column(df,column_name):
    if column_name=="gender":

        #gender
        for col in df.columns:
            df[col]=df[col].replace("Female",0)
            df[col]=df[col].replace("Male",1)

    elif column_name=="multi_screen":

        #multi_screen
        df['multi_screen'] = df['multi_screen'].apply(lambda x: str(x).replace('no', "0"))
        df['multi_screen'] = df['multi_screen'].apply(lambda x: str(x).replace('yes', "1"))
        df['multi_screen'] = df['multi_screen'].astype(int)


    elif column_name=="mail_subscribed":

        #mail_subscribed
        df["mail_subscribed"]=df.mail_subscribed.map({"no":0,"yes":1})

    else:
        print("Lütfen categoric colon isimlerinden birini giriniz.(gender,multi_screen,mail_subscribed)")    


    df.info()



def independent_dependent(df):
    y=df["churn"]
    X=df.drop("churn",axis=1)
    print(X.head())
    print(y.head())
    return X,y