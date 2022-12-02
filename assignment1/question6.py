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


tuning_logreg = pd.read_csv("results/tuning_logreg.csv")
print(tuning_logreg.head())