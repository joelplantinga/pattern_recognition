import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt


def has_double(row):

    m = row!=0 # indexes of non zero numbers
    return (np.count_nonzero(m[1:] > m[:-1]) + m[0]) > 1

def check_doubles(img):
    
    img = img.reshape(28, 28)
    return len([row for row in img if has_double(row)])
    

def create_double_att(data, labels, show_plot=False):
    # of the number of rows that have two lines with a gap in between
    doubles = np.array([check_doubles(img) for img in data])
    doubles_mean = [np.mean(doubles[labels == i]) for i in range(10)]
    doubles_stdv = [np.std(doubles[labels == i]) for i in range(10)]
    
    data = pd.DataFrame(data={'digit': list(range(0, 10)), 'doubles_mean': doubles_mean, 'doubles_std': doubles_stdv})
    data.to_csv('results/doubles_data.csv', index=False)

    if show_plot:
        plt.bar(range(10), doubles_mean, yerr = doubles_stdv)
        plt.show()
    return doubles

    



if __name__ == '__main__':

    """Point 3: new feature"""

    # number of pixels
    # nr_pixels = np.asarray([ len([pix for pix in row if pix != 0]) for row in digits])
    # nr_pixels = scale(nr_pixels).reshape(-1,1)

    """" Read the data"""
    mnist_data = pd.read_csv('mnist.csv').values
    labels = mnist_data[:, 0]
    digits = mnist_data[:, 1:]
    img_size = 28
    
    
    n = 3795
    mask = np.hstack([np.random.choice(np.where(labels == l)[0], n, replace=False) for l in np.unique(labels)])

    labels = mnist_data[mask, 0]
    digits = mnist_data[mask, 1:]


    doubles = create_double_att(digits, labels, show_plot=False).reshape(-1, 1)

    
    density_data = pd.DataFrame(data={'doubles': doubles.flatten(), 'digit': labels})
    density_data["value"]=1
    density_data = density_data.pivot_table(index='doubles', columns='digit', values='value', aggfunc=len, fill_value=0)
    density_data.to_csv('results/density_double.csv')

    exit(0)
    # data = [doubles[labels == i,] for i in [0,2,3,4,5,6,7,8,9]]

    # ax = sns.violinplot(data=data)
    # ax.set_xticklabels([0,2,3,4,5,6,7,8,9])
    # plt.show()

    # for i in range(10):
    #     print(len(doubles[labels == 0]))

    # print(doubles.shape)

    # logistic regression with feature doubles
    model = LogisticRegression(random_state=0, class_weight = 'balanced').fit(doubles, labels)
    labels_predicted = model.predict(doubles)


    new_data = pd.DataFrame(data={'doubles': doubles.flatten(), 'digit': labels_predicted})

    new_data["value"]=1
    print(new_data.head())


    new_data = new_data.pivot_table(index='doubles', columns='digit', values='value', aggfunc=len, fill_value=0)#.plot(kind='line')

   
    new_data.to_csv('results/prediction_results_per_double.csv')


    conf_mat = confusion_matrix(labels, labels_predicted)
    np.savetxt("results/confusion_matrix_doubles.csv", conf_mat, delimiter=",")
    print(confusion_matrix(labels, labels_predicted))
    df = pd.DataFrame(confusion_matrix(labels, labels_predicted))
    df.to_csv('results/confusion_matrix_doubles.csv')


    # need to add the analysis similar to ink (mean and sd)