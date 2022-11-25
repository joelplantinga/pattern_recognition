import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix, accuracy_score


"""" Read the data"""
mnist_data = pd.read_csv('mnist.csv').values
labels = mnist_data[:, 0]
digits = mnist_data[:, 1:]
img_size = 28
# plt.imshow(digits[0].reshape(img_size, img_size))
# # plt.show()


"""Point 4: fit the new model and compare with previous one"""
# number of pixels
# of the number of rows that have two lines with a gap in between
nr_pixels = np.asarray([ len([pix for pix in row if pix != 0]) for row in digits])
nr_pixels = scale(nr_pixels).reshape(-1,1)
ink = np.array([sum(row) for row in digits])
ink = scale(ink).reshape(-1, 1) # reshape makes it a column vector
new_features = np.hstack((ink, nr_pixels))
model = LogisticRegression( random_state=0).fit(new_features, labels)
labels_predicted = model.predict(new_features)
print(confusion_matrix(labels, labels_predicted))
print(f'Accuracy of model with ink and nr_pixels features: {accuracy_score(labels, labels_predicted)}')

