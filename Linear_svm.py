import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_wine
from sklearn.svm import LinearSVC
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

    if do_plot:
        df = pd.DataFrame(X)
        colors = get_colors(t)
        pd.plotting.scatter_matrix(df,color=colors,diagonal='kde')
        #corr = X.corr()
        #sns.hitmap(corr)

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

def do_svm(tr_X, tr_t, te_X, te_t,title):

    model = LinearSVC()
    model.fit(tr_X,tr_t)

    te_t_pred = model.predict(te_X)
    cm = confusion_matrix(te_t,te_t_pred)
    score = model.score(te_X, te_t)
    if score >= 0.85:
        print('[*] do svm(%s)' % title)
        #print('[-] row: actual, cols: predict')
        #print(cm)

        print('[-] accuracy : ', score)

def main():
    do_plot = True
    tr_X, tr_t, te_X, te_t = load_data(do_plot)

    #do_svm(tr_X, tr_t, te_X, te_t,'using all features')

    for i in range(tr_X.shape[1]):
        for j in range(tr_X.shape[1]):
            cols = [i,j] if i != j else [i]
            tr_X_ex = tr_X[:,cols]
            te_X_ex = te_X[:,cols]

            do_svm(tr_X_ex, tr_t, te_X_ex, te_t,'using %s-th features'% str((cols)))

    if do_plot:
        plt.show()

if __name__ == '__main__':
    main()