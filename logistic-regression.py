import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

def generate_data(m,S,size,cls):
    print("[*] generate data(class %d)"% cls)
    X = np.random.multivariate_normal(m,S,size)
    t = np.ones((size)) * cls

    return X, t

def load_data(do_plot=False):
    print('[*] load data')
    X_1, t_1 = generate_data([1, 1], [[12, 0], [0, 1]], 1000, 1)
    X_2, t_2 = generate_data([7, 7], [[8, 3], [3, 2]], 1000, 2)
    X_3, t_3 = generate_data([15, 1], [[2, 0], [0, 2]], 1000, 3)
    if do_plot:
        print('[-] do plot')

        fig = plt.figure()
        plt.scatter(X_1[:, 0], X_1[:, 1], c='r',s=3, label='class 1')
        plt.scatter(X_2[:, 0], X_2[:, 1], c='g',s=3, label='class 2')
        plt.scatter(X_3[:, 0], X_3[:, 1], c='b',s=3, label='class 3')

        plt.legend()
        fig.show()
    X = np.concatenate((X_1,X_2,X_3),0)
    t = np.concatenate((t_1,t_2,t_3),0)

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

def do_logistic_regression(tr_X, tr_t, te_X, te_t, do_plot=False):
    print('[*] do logistic regression')

    model = LogisticRegression()

    print('[-] fit model...')
    model.fit(tr_X, tr_t)

    te_t_pred = model.predict(te_X)

    score = model.score(te_X, te_t)
    print('[-] accuracy :',score)

    if do_plot:
        print('[-] do plot')
        x_space = np.linspace(-15,20,10)
        y_space = np.linspace(-4, 12, 10)
        X, Y = np.meshgrid(x_space,y_space)
        XY = np.dstack((X,Y)).reshape((-1,2))
        prob = model.predict_proba(XY).reshape((10, 10, 3))
        #print(prob.shape) # probability of each class

        ind_t1 = np.where(te_t_pred == 1.)[0]
        ind_t2 = np.where(te_t_pred == 3.)[0]
        ind_t3 = np.where(te_t_pred == 2.)[0]

        te_X_pred1 = te_X[ind_t1]
        te_X_pred2 = te_X[ind_t2]
        te_X_pred3 = te_X[ind_t3]

        fig = plt.figure()
        plt.imshow(prob, extent=(-15,20,-4,12),cmap='jet',alpha=0.5,origin='lower')
        plt.scatter(te_X_pred1[:, 0], te_X_pred1[:, 1], c='r',s=3, label='class 1 pred')
        plt.scatter(te_X_pred2[:, 0], te_X_pred2[:, 1], c='g',s=3, label='class 2 pred')
        plt.scatter(te_X_pred3[:, 0], te_X_pred3[:, 1], c='b',s=3, label='class 3 pred')
        plt.legend()
        fig.show()



def main():
    do_plot = True
    tr_X, tr_t, te_X, te_t = load_data(do_plot)
    do_logistic_regression(tr_X, tr_t, te_X, te_t, do_plot)

    if do_plot:
        plt.show()

if __name__ == '__main__':
    main()