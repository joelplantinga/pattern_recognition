import numpy as np





import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix, accuracy_score
from question3 import create_double_att
from question2 import create_ink_feature




if __name__ == '__main__':


    """" Read the data"""
    mnist_data = pd.read_csv('mnist.csv').values

    labels = mnist_data[:, 0]
    digits = mnist_data[:, 1:]
    
    n = 3795
    mask = np.hstack([np.random.choice(np.where(labels == l)[0], n, replace=False) for l in np.unique(labels)])


    labels = mnist_data[mask, 0]
    digits = mnist_data[mask, 1:]

    print(labels.shape)
    print(digits.shape)

    img_size = 28

    """Point 4: fit the new model and compare with previous one"""
    # number of pixels
    # nr_pixels = np.asarray([ len([pix for pix in row if pix != 0]) for row in digits])
    # nr_pixels = scale(nr_pixels).reshape(-1,1)
    # ink = np.array([sum(row) for row in digits])

    ink = create_ink_feature(digits, labels)
    ink = scale(ink).reshape(-1, 1) # reshape makes it a column vector

    doubles = create_double_att(digits, labels)
    new_features = np.hstack((ink, doubles.reshape(-1, 1)))


    model = LogisticRegression().fit(new_features, labels)
    labels_predicted = model.predict(new_features)
    print(confusion_matrix(labels, labels_predicted))

    df = pd.DataFrame(confusion_matrix(labels, labels_predicted))
    df.to_csv('results/confusion_matrix_both.csv')


    print(f'Accuracy of model with gapped doubles and nr_pixels features: {accuracy_score(labels, labels_predicted)}')




