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



"""" Read the data"""
mnist_data = pd.read_csv('mnist.csv').values
labels = mnist_data[:, 0]
digits = mnist_data[:, 1:]
img_size = 28
# plt.imshow(digits[0].reshape(img_size, img_size))
# plt.show()

reshaped_digits = np.array([rescale_img(row) for row in digits])

index = np.random.choice(42000, 5000, replace=False)
X_train = scale(reshaped_digits[index,:])
Y_train = labels[index] 
X_test = scale(np.delete(reshaped_digits, index, axis=0))
Y_test = np.delete(labels, index, axis=0)

# logistic = LogisticRegression(solver= 'liblinear', random_state=0, penalty='l1')
# C = np.arange(1, 1000, 1)
# distributions = dict(C=C)
# clf = RandomizedSearchCV(logistic, distributions, random_state=0)
# search = clf.fit(X_train, Y_train)
# print(search.best_params_)

def objective(trial):

    C = trial.suggest_int("C", 1, 101, 10)

    logistic = LogisticRegression(solver= 'liblinear', random_state=0, C=C, penalty='l1')

    score = cross_val_score(logistic, X_train, Y_train, n_jobs=-1)
    accuracy = score.mean()

    return accuracy


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=200)
# found_param = study.trials_dataframe()
# found_param.to_csv(f"results/preprocessing_{CLASSIFIER}_{NGRAM}_cv.csv")

print(study.best_trial)


