from sklearn.model_selection import train_test_split

def split_data_test(X,y):
    """
        Data X ve y şekinde olmalıdır.
    """
    #Train=train+val   test   80-20
    X_train, x_test, Y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("X_train:", X_train.shape)
    print("x_test:", x_test.shape)
    print("Y_train:", Y_train.shape)
    print("y_test:", y_test.shape)
    return X_train, x_test, Y_train, y_test

def split_data_val(X_train,Y_train):
    # train-val    60-20
    """
        split_data_test den gelen X_train,Y_train verilmelidir.
    """
    x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.25, random_state=42)
    print("x_train:", x_train.shape)
    print("x_val:", x_val.shape)
    print("y_train:", y_train.shape)
    print("y_val:", y_val.shape)
    return  x_train, x_val, y_train, y_val

