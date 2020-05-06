import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
import warnings

warnings.filterwarnings("ignore") # warning 무시

def get_colors(t):
    color_wheel ={
        0: '#0000ff',
        1: '#7bc043',
        2: '#ff0000'
    }
    colors = list()
    for i in range(len(t)):
        colors.append(color_wheel[t[i]])
    return np.array(colors)

def load_data(T, do_plot=False):
    print('[*] load data')
    if T == 'wine':
        X, t = load_wine(return_X_y=True)
        if do_plot:
            df = pd.DataFrame(X)
            colors = get_colors(t)
            pd.plotting.scatter_matrix(df,color=colors,diagonal='kde')
        X = StandardScaler().fit_transform(X)  # 정규화
    elif T == 'mnist':
        X, t = load_digits(return_X_y=True)
        #X, t = fetch_openml('mnist_784',version=1,return_X_y=True)
        #X = X / 255
        #plt.imshow(X[0].reshape(8,8))
        #plt.show()

    ind = np.arange(len(X))
    np.random.shuffle(ind)
    X = X[ind]
    t = t[ind]

    tr_size = int(len(X)*3/4)
    tr_X = X[:tr_size]
    tr_t = t[:tr_size]
    te_X = X[tr_size:]
    te_t = t[tr_size:]
    #tr_X = StandardScaler().fit_transform(tr_X)
    #te_X = StandardScaler().fit_transform(te_X)
    print('success')
    return tr_X, tr_t, te_X, te_t

def do_logistic_regression(tr_X, tr_t, te_X, te_t,title):
    print('[*] do logistic_regression(%s)'%title)

    model = LogisticRegression()
    model.fit(tr_X,tr_t)

    te_t_pred = model.predict(te_X)
    CM = confusion_matrix(te_t,te_t_pred)
    print('[-] row: actual, cols: predict')
    print(CM)
    print("Class 1 : ", np.round(model.coef_[0,:],3))  # 상관계수
    print("Class 2 : ", np.round(model.coef_[1, :], 3))
    print("Class 3 : ", np.round(model.coef_[2, :], 3))
    score = model.score(te_X,te_t)
    print('[-] accuracy : ', score)
    print()

    return model

def do_Decision_Tree_Classifier(tr_X, tr_t, te_X, te_t, title):
    print('[*] do Decision_Tree_Classifier(%s)' % title)
    model = DecisionTreeClassifier()
    model.fit(tr_X, tr_t)


    score = model.score(te_X, te_t)
    print('[-] accuracy : ', score)
    print('feature_importances : \n',model.feature_importances_)
    #print('',model.tree_.max_depth)
    print()

    tet_pred  = model.predict(te_X)
    return model, score

def DTC_analyze():
    score = 0
    DTC_result = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(0, 100):
        tr_X, tr_t, te_X, te_t = load_data('wine',False) # train set 새로 추출(랜덤 추출 다시)
        DTC, P_score = do_Decision_Tree_Classifier(tr_X, tr_t, te_X, te_t, 'using all features ' + str(i) + '  ')
        DTC_result += DTC.feature_importances_ * P_score # score로 weight 적용
        score = score + P_score # 나중에 score들의 합으로 나누기 위해

    DTC_result = DTC_result / score # 평균치 계산

    print('평균 Score : ', score)
    print('평균 feature importances : ', DTC_result)

    x = list(range(13))
    plt.figure(1)
    plt.bar(x, DTC_result)
    plt.xlabel('feature')
    plt.ylabel('importance')

    plt.title('importances in Decision tree')
    return DTC_result


def do_Perceptron(tr_X, tr_t, te_X, te_t , tuple):
    model =  MLPClassifier(hidden_layer_sizes=tuple)

    model.fit(tr_X,tr_t)
    score = model.score(te_X,te_t)
    print(score)

    te_t_pred = model.predict(te_X)
    CM = confusion_matrix(te_t, te_t_pred)
    print('[-] row: actual, cols: predict')
    print(CM)
    plt.figure()
    fig, axes = plt.subplots(4, 4)
    vmin, vmax = model.coefs_[0].min(), model.coefs_[0].max()
    for coef, ax in zip(model.coefs_[0].T, axes.ravel()):
        ax.matshow(coef.reshape(8, 8), cmap=plt.cm.gray, vmin=.5 * vmin,
                   vmax=.5 * vmax)
        ax.set_xticks(())
        ax.set_yticks(())




def main():
    do_plot = True
    tuple = (200,200,200)
    tr_X, tr_t, te_X, te_t = load_data('mnist',do_plot)
    do_Perceptron(tr_X, tr_t, te_X, te_t,tuple)

    tr_X, tr_t, te_X, te_t = load_data('wine',do_plot)
    do_logistic_regression(tr_X, tr_t, te_X, te_t,'using all features')

    DTCresult = DTC_analyze() 
    important_feature = []
    for i in range(len(DTCresult)):
        if DTCresult[i] >= 0.2: # feature importance 가 0.2 이상인 feature 추출
            important_feature.append(i)
    print(important_feature)

    tr_X_ex = tr_X[:, important_feature]
    te_X_ex = te_X[:, important_feature]

    model,score = do_Decision_Tree_Classifier( tr_X_ex, tr_t, te_X_ex, te_t, 'using '+str(important_feature)+ 'features')
    plt.figure()
    plot_tree(model)

    if do_plot:
        plt.show()

if __name__ == '__main__':
    main()