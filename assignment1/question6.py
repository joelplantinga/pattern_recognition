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

mnist_data = delete_constant_columns(mnist_data)
# print(mnist_data.shape)
index = np.random.choice(42000, 5000, replace=False)
X_train, y_train, X_test, y_test = split_data(mnist_data, labels, index)


np.random.seed(0)
tuning_logreg = pd.read_csv("results/tuning_logreg.csv")
best_C = tuning_logreg.loc[np.argmax(tuning_logreg['value']), 'params_C']

logreg_model = LogisticRegression(C=best_C, solver='saga', max_iter=250, penalty='l1')
logreg_model.fit(X_train, y_train)

y_predict = logreg_model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_predict)}")
print(confusion_matrix(y_test, y_predict))



