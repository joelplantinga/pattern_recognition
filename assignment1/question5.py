import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score
import cv2


def rescale_img(img):

    img = img.reshape(28, 28)
    img = cv2.resize(img.astype(float), (14, 14), interpolation=cv2.INTER_CUBIC)
    return img.flatten()

def delete_constant_columns(data):

    data = data.transpose()
    data = [row for row in data if len(np.unique(row)) > 1]
    return np.asarray(data).transpose()

def split_data(data, labels, index):

    X_train = data[index,:]
    y_train = labels[index] 
    X_test = np.delete(data, index, axis=0)
    y_test = np.delete(labels, index, axis=0)
    
    return X_train, y_train, X_test, y_test
    

"""" Read the data"""
mnist_data = pd.read_csv('mnist.csv').values
labels = mnist_data[:, 0]
digits = mnist_data[:, 1:]
img_size = 28

# mnist_data = np.array([rescale_img(row) for row in digits])


# print(pd.DataFrame(mnist_data).describe()) 
# print(mnist_data.shape)

mnist_data = delete_constant_columns(mnist_data)

print(mnist_data.shape)

index = np.random.choice(42000, 5000, replace=False)

X_train, y_train, X_test, y_test = split_data(mnist_data, labels, index)

# logistic = LogisticRegression(solver= 'liblinear', random_state=0, penalty='l1')
# C = np.arange(1, 1000, 1)
# distributions = dict(C=C)
# clf = RandomizedSearchCV(logistic, distributions, random_state=0)
# search = clf.fit(X_train, Y_train)
# print(search.best_params_)

def objective(trial):

    C = trial.suggest_float("C", 0.001, 1000, log=True)

    logistic = LogisticRegression(solver= 'liblinear', random_state=0, C=C, penalty='l1')
    

    # !! cross validation is called without shuffling on default in order to get the same results for every call
    score = cross_val_score(logistic, X_train, y_train, n_jobs=-1)
    accuracy = score.mean()
    return accuracy


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)
found_param = study.trials_dataframe()
found_param.to_csv(f"results/tuning_logreg.csv")

print(study.best_trial)


