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
from sklearn.preprocessing import StandardScaler
from question5 import rescale_img, delete_constant_columns, split_data    



if __name__ == "__main__":

    np.random.seed(0)
    np.set_printoptions(precision=3)


    """" Read the data"""
    mnist_data = pd.read_csv('mnist.csv').values
    labels = mnist_data[:, 0]
    digits = mnist_data[:, 1:]

    mnist_data = np.array([rescale_img(row) for row in digits])

    mnist_data = delete_constant_columns(mnist_data)
    index = np.random.choice(42000, 5000, replace=False)


    X_train, y_train, X_test, y_test = split_data(mnist_data, labels, index)

    print(f"train distribution: {np.bincount(y_train) / 5000}")
    print(f"test distribution: {np.bincount(y_test) / 37000}")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    """Fit the Logistic Regression with best parameter setting """

    # tuning_logreg = pd.read_csv("results/tuning_logreg.csv")
    # best_C = tuning_logreg.loc[np.argmax(tuning_logreg['value']), 'params_C']

    # logreg_model = LogisticRegression(C=best_C, solver='saga', max_iter=700, penalty='l1', tol=0.001)
    # logreg_model.fit(X_train, y_train)

    # y_predict = logreg_model.predict(X_test)
    # print("------ Logistic Regression --------")
    # print(f"best parameter setting C={best_C}")

    # print(f"Accuracy: {accuracy_score(y_test, y_predict)}")
    # print(confusion_matrix(y_test, y_predict))


    """ Fit the SVM with best parameter setting """

    tuning_svm = pd.read_csv("results/tuning_svm1.csv")
    best_C = tuning_svm.loc[np.argmax(tuning_svm['value']), 'params_C']
    best_gamma = tuning_svm.loc[np.argmax(tuning_svm['value']), 'params_gamma']

    svm_model = SVC(C=best_C, gamma=0.00000001, class_weight='balanced')
    svm_model.fit(X_train, y_train)

    y_pred = svm_model.predict(X_test)
    print("------ Support Vector Machines --------")
    print(f"best parameter setting C={best_C}, gamma={best_gamma}")
    print(f"Accuracy of SVM: {accuracy_score(y_test, y_pred)}")
    print(confusion_matrix(y_test, y_pred))

