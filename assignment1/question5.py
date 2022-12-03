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
from sklearn.svm import SVC


np.random.seed(0)

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

mnist_data = np.array([rescale_img(row) for row in digits])


mnist_data = delete_constant_columns(mnist_data)

index = np.random.choice(42000, 5000, replace=False)

X_train, y_train, X_test, y_test = split_data(mnist_data, labels, index)


def objective(trial):

    C = trial.suggest_float("C", 0.001, 10000, log=True)

    # changed the solver to saga since it's the only one that supports l1 and multiclass problem
    logistic = LogisticRegression(solver= 'saga', random_state=0, C=C, penalty='l1', max_iter= 700, tol=0.001)

    # !! cross validation is called without shuffling on default in order to get the same results for every call
    score = cross_val_score(logistic, X_train, y_train, n_jobs=-1)
    accuracy = score.mean()
    return accuracy


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)
found_param = study.trials_dataframe()
found_param.to_csv(f"results/tuning_logreg.csv")
print(" Logistic Regression ")
print(study.best_trial)




def objectiveSVM(trial):

    C = trial.suggest_float("C", 100000, 10000000, log=True)
    gamma = trial.suggest_float("gamma", 0.0000001, 0.00001, log=True)

    support_vector_machines = SVC(C=C, gamma=gamma)

    # !! cross validation is called without shuffling on default in order to get the same results for every call
    score = cross_val_score(support_vector_machines, X_train, y_train, n_jobs=-1)

    accuracy = score.mean()
    return accuracy



studySVM = optuna.create_study(direction="maximize")
studySVM.optimize(objectiveSVM, n_trials=100)
found_paramSVM = studySVM.trials_dataframe()
found_paramSVM.to_csv(f"results/tuning_SVM.csv")
print(" Support Vector Machines ")
print(studySVM.best_trial)

