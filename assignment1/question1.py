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
# plt.show()


""" Point 1: exploratory analysis"""
# describe can be used only on a dataFrame not on array 
print(pd.DataFrame(mnist_data).describe())
# # we can eliminate those variables where min and max coincide, because it means that for all observation that pixel is the same


plt.hist(labels, bins=10) 
plt.show() 
print(f'Percentage of correct predictions assigning all to class 1: {len(labels[labels==1])/len(labels)}')
# evenly distributed 
