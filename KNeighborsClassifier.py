import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_wine
from sklearn.neighbors import KNeighborsClassifier
import warnings

warnings.filterwarnings("ignore") # warning 무시

def get_colors(t):
    color_wheel ={
        0: '#0392cf',
        1: '#7bc043',
        2: '#ee4035'
    }
    colors = list()
    for i in range(len(t)):
        colors.append(color_wheel[t[i]])
    return np.array(colors)

def load_data(do_plot=False):
    print('[*] load data')
    X, t = load_wine(return_X_y=True)

    ind = np.arange(len(X))
    np.random.shuffle(ind)
    X = X[ind]
    t = t[ind]

    tr_size = int(len(X)*3/4)
    tr_X = X[:tr_size]
    tr_t = t[:tr_size]
    te_X = X[tr_size:]
    te_t = t[tr_size:]

    return tr_X, tr_t, te_X, te_t

def do_KNN(k,tr_X, tr_t, te_X, te_t,title):
    print('[*] do svm(%s)'%title)

    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(tr_X,tr_t)
    score = classifier.score(te_X,te_t)
    print('[-] accuracy : ', score)
    print()
    return score

def main():
    accuracies = []
    do_plot = True
    tr_X, tr_t, te_X, te_t = load_data(do_plot)
    tr_X_ex = tr_X[:, [9, 11]]
    te_X_ex = te_X[:, [9, 11]]
    i=1
    while i <= 100:
        accuracies.append(do_KNN(i,tr_X_ex, tr_t, te_X_ex, te_t,'Nu :'+str(i) ))
        i = i+1

    plt.plot(range(1,101),accuracies)
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.title("KNN")


    if do_plot:
        plt.show()

if __name__ == '__main__':
    main()