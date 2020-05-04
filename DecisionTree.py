import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
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

def load_data(do_plot=False):
    print('[*] load data')
    X, t = load_wine(return_X_y=True)
    X = StandardScaler().fit_transform(X) # 정규화
    if do_plot:
        plt.figure()
        df = pd.DataFrame(X)
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
    #tr_X = StandardScaler().fit_transform(tr_X)
    #te_X = StandardScaler().fit_transform(te_X)
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
    print('',model.tree_.max_depth)
    print()

    tet_pred  = model.predict(te_X)
    return model, score

def DTC_analyze():
    score = 0
    DTC_result = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(0, 100):
        tr_X, tr_t, te_X, te_t = load_data(False) # train set 새로 추출(랜덤 추출 다시)
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


def main():
    do_plot = True
    tr_X, tr_t, te_X, te_t = load_data(do_plot)

    LR = do_logistic_regression(tr_X, tr_t, te_X, te_t,'using all features')

    DTCresult = DTC_analyze() # 운좋게 좋은 trainset이 추출되거나 학습되는 것을 방지하기 위해 100번 반복하여 평균값 출력
    important_feature = []
    for i in range(len(DTCresult)):
        if DTCresult[i] >= 0.1: # feature importance 가 0.1 이상인 feature 추출
            important_feature.append(i)
    print(important_feature)

    tr_X_ex = tr_X[:, important_feature]
    te_X_ex = te_X[:, important_feature]

    do_Decision_Tree_Classifier( tr_X_ex, tr_t, te_X_ex, te_t, 'using '+str(important_feature)+ 'features')


    if do_plot:
        plt.show()

if __name__ == '__main__':
    main()