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
plt.imshow(digits[0].reshape(img_size, img_size))
# plt.show()


""" Point 1: exploratory analysis"""
# describe can be used only on a dataFrame not on array 
# print(mnist_data.describe()) 
# # we can eliminate those variables where min and max coincide, because it means that for all observation that pixel is the same


plt.hist(labels, bins=10) 
# plt.show() 
print(f'Percentage of correct predictions assigning all to class 1: {len(labels[labels==1])/len(labels)}')
# evenly distributed 


"""" Point 2: ink feature and logistic regression """
# create ink feature
ink = np.array([sum(row) for row in digits])
# compute mean for each digit class
ink_mean = [np.mean(ink[labels == i]) for i in range(10)]
# compute standard deviation for each digit class
ink_std = [np.std(ink[labels == i]) for i in range(10)]

for i in range(10):
    print(f'------- Ink for digit {i} ------ \n  Mean: {ink_mean[i]} \n SD: {ink_std[i]}')

# plotting mean_ink with the corresponding standard deviation 
plt.bar(range(10), ink_mean, yerr = ink_std)
# plt.show()


# logistic regression with feature ink
ink = scale(ink).reshape(-1, 1) # reshape makes it a column vector
model = LogisticRegression(random_state=0).fit(ink, labels)
labels_predicted = model.predict(ink)
print(confusion_matrix(labels, labels_predicted))

"""Point 3: new feature and new model"""
# number of pixels
# of the number of rows that have two lines with a gap in between
nr_pixels = [ len([pix for pix in row if pix != 0]) for row in digits]

new_features = np.hstack((ink, nr_pixels))
model = LogisticRegression(random_state=0).fit(ink, labels)
labels_predicted = model.predict(ink)
print(confusion_matrix(labels, labels_predicted))