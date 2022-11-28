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

# # Distirbution of digits: histogram
# plt.hist(labels, bins=10) 
# plt.show() 


# # Frequency table
occurency_table = np.empty(shape=(10, 3), dtype=object)

for i in range(len(np.unique(labels))):
    count = np.count_nonzero(labels==i)
    percentage = (count/len(digits)*100)
    occurency_table[i]= int(i), int(count), round(float(percentage),2)

print(occurency_table)


# # Digits are quite evenly distibuted, meaning that predicting the majority class will leave us with a low succes rate

print(f'Percentage of correct predictions assigning all to class 1: {len(labels[labels==1])/len(labels)}')
# evenly distributed 

