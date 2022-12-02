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
np.random.seed(0)
index = np.random.choice(42000, 5000, replace=False)

X_train, y_train, X_test, y_test = split_data(mnist_data, labels, index)

# logistic = LogisticRegression(solver= 'saga', random_state=0, penalty='l1')
# C = np.arange(1, 1000, 1)
# distributions = dict(C=C)
# clf = RandomizedSearchCV(logistic, distributions, random_state=0)
# search = clf.fit(X_train, y_train)
# print(search.best_params_)

def objective(trial):

    C = trial.suggest_float("C", 0.001, 1000, log=True)

    # changed the solver to saga since it's the only one that supports l1 and multiclass problem
    logistic = LogisticRegression(solver= 'saga', random_state=0, C=C, penalty='l1', max_iter= 1000)

    # !! cross validation is called without shuffling on default in order to get the same results for every call
    score = cross_val_score(logistic, X_train, y_train, n_jobs=-1)
    accuracy = score.mean()
    return accuracy


# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=100)
# found_param = study.trials_dataframe()
# found_param.to_csv(f"results/tuning_logreg.csv")

# print(study.best_trial)

def objectiveSVM(trial):

    C = trial.suggest_float("C", 1, 1000, log=True)
    gamma = trial.suggest_float('gamma', 0.001, 1000)

    support_vector_machines = SVC(C=C, gamma=gamma)

    # !! cross validation is called without shuffling on default in order to get the same results for every call
    score = cross_val_score(support_vector_machines, X_train, y_train, n_jobs=-1)

    accuracy = score.mean()
    return accuracy



studySVM = optuna.create_study(direction="maximize")
studySVM.optimize(objectiveSVM, n_trials=100)
found_paramSVM = studySVM.trials_dataframe()
found_paramSVM.to_csv(f"results/tuning_SVM.csv")

print(studySVM.best_trial)



"""randomized search"""
# SVM = SVC()
# C = np.arange(1, 10, 1)
# gamma = np.arange(1, 10, 1)
# distributions = dict(C=C, gamma=gamma)
# clf = RandomizedSearchCV(SVM, distributions, refit=True, random_state=0, scoring='accuracy')
# model = clf.fit(X_train, y_train)
# print(f" best estimator: {model.best_estimator_}, best score: {model.best_score_} ")