import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix, accuracy_score



def create_ink_feature(data, labels, show_plot=False):

    # create ink feature
    ink = np.array([sum(row) for row in data])
    # compute mean for each digit class
    ink_mean = [np.mean(ink[labels == i]) for i in range(10)]
    # compute standard deviation for each digit class
    ink_std = [np.std(ink[labels == i]) for i in range(10)]

    if show_plot:
        plt.bar(range(10), ink_mean, yerr = ink_std)
        plt.show()
    
    return ink


if __name__ == '__main__':

    """" Read the data"""
    mnist_data = pd.read_csv('mnist.csv').values
    labels = mnist_data[:, 0]
    digits = mnist_data[:, 1:]
    img_size = 28


    """" Point 2: ink feature and logistic regression """



    ink = create_ink_feature(digits, labels, show_plot=True)
    # logistic regression with feature ink
    ink = scale(ink).reshape(-1, 1) # reshape makes it a column vector
    model = LogisticRegression(random_state=0).fit(ink, labels)
    labels_predicted = model.predict(ink)
    print(confusion_matrix(labels, labels_predicted))
    print(f'Accuracy of model with ink feature: {accuracy_score(labels, labels_predicted)}')

