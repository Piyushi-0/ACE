import numpy as np
import sklearn.model_selection
import joblib


def get_data(num_of_data_points):
    coin_flip = np.random.binomial(1,0.5,num_of_data_points)

    X = []
    y = []
    for i in range(num_of_data_points):
        T = np.random.randint(100,110)
        feature_vector = np.random.normal(0.0,0.2,T)
        if coin_flip[i] == 1:
            feature_vector[:3] = 1.0 #+ np.random.normal(0.0,0.2,3)
            y.append(1.0)
        else:
            feature_vector[:3] = -1.0 #+ np.random.normal(0.0,0.2,3)
            y.append(0.0)
        
        X.append(feature_vector)
    print len(X)
    train_x, test_x = sklearn.model_selection.train_test_split(np.array(X), test_size = 2560, random_state = 42)
    train_y, test_y = sklearn.model_selection.train_test_split(np.array(y), test_size = 2560, random_state = 42)
    return train_x, test_x, train_y, test_y


train_x, test_x, train_y, test_y = get_data(1000000)

joblib.dump(train_x, "train_x.pkl")
joblib.dump(train_y, "train_y.pkl")
joblib.dump(test_x, "test_x.pkl")
joblib.dump(test_y, "test_y.pkl")



