import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_wine
from sklearn.svm import NuSVC
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
        df = pd.DataFrame(X[:,[9,11]])
        colors = get_colors(t)
        pd.plotting.scatter_matrix(df,color=colors,diagonal='kde')

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

def do_nu_svm(Nu,tr_X, tr_t, te_X, te_t,title):
    print('[*] do svm(%s)'%title)

    model = NuSVC(nu=float(Nu))
    model.fit(tr_X,tr_t)

    te_t_pred = model.predict(te_X)
    cm = confusion_matrix(te_t,te_t_pred)
    #print('[-] row: actual, cols: predict')
    #print(cm)

    score = model.score(te_X,te_t)
    print('[-] accuracy : ', score)
    print()

def main():
    do_plot = True
    tr_X, tr_t, te_X, te_t = load_data(do_plot)
    tr_X_ex = tr_X[:, [9, 11]]
    te_X_ex = te_X[:, [9, 11]]
    i=0.1
    while i <= 0.8:
        i=round(i,1)
        try:
            do_nu_svm(i,tr_X_ex, tr_t, te_X_ex, te_t,'Nu :'+str(i) )
        except:
            break
        i=i+0.1



    if do_plot:
        plt.show()

if __name__ == '__main__':
    main()