import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix


"""" Read the data"""
mnist_data = pd.read_csv('mnist.csv').values
labels = mnist_data[:, 0]
digits = mnist_data[:, 1:]
img_size = 28
# plt.imshow(digits[0].reshape(img_size, img_size))
# # plt.show()


"""Point 3: new feature"""
# number of pixels
# of the number of rows that have two lines with a gap in between
nr_pixels = np.asarray([ len([pix for pix in row if pix != 0]) for row in digits])
nr_pixels = scale(nr_pixels).reshape(-1,1)
ink = np.array([sum(row) for row in digits])
ink = scale(ink).reshape(-1, 1) # reshape makes it a column vector
new_features = np.hstack((ink, nr_pixels))

# need to add the analysis similar to ink (mean and sd)