import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from sklearn.model_selection import train_test_split
import random
from keras import backend as K
import warnings
from sklearn.metrics import confusion_matrix
warnings.filterwarnings("ignore") # warning 무시


class load_data:
    def __init__(self):
        self.X = []
        self.Y = []
        self.path = '../crop_part1/'
        self.image_list = os.listdir(self.path)

        # suffle
        random.shuffle(self.image_list)

        for i in range(len(self.image_list)):
            im = Image.open(self.path+self.image_list[i])
            im = im.convert("RGB")
            data = np.asarray(im)
            try:
                #self.Y.append(int(self.image_list[i].split('_')[2]))
                P_label = []
                temp = self.image_list[i].split('_')
                P_label.append(int(temp[1]))
                P_label.append(int(temp[2]))
                self.Y.append(P_label)
                self.X.append(data)

            except Exception as e:
                print(e)

        self.X = np.array(self.X)
        self.Y = np.array(self.Y)

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=0.1)
        xy = (self.X_train, self.X_test, self.Y_train, self.Y_test)
        np.save("./image_data.npy",xy)

class Custom_CNN:
    def __init__(self):
        self.model = tf.keras.Sequential(self._build_layers())
        #self.model.compile(optimizer='adam',  loss=self._gender loss, metrics=[self._gender_accuracy])
        #self.model.compile(optimizer='adam',  loss=self._race_loss , metrics=[self._race_accuracy])
        self.model.compile(optimizer='adam',  loss=self._sum_loss , metrics=[self._gender_accuracy, self._race_accuracy, self.precision,self.recall,self.f1score])

    def _sum_loss(self, y_true, y_pred):
        return (self._gender_loss(y_true,y_pred)+self._race_loss(y_true,y_pred))/2.

    def _gender_loss(self, y_true, y_pred):
        # (7822, 2) 0 인덱스는 성별 1인덱스는 인종
        y_true = tf.split(y_true, 2, 1)  # 분리!
        y_true = tf.cast(y_true[0], tf.float32)  # (7822 , 1)
        y_pred = tf.split(y_pred,6,1)[0]

        y_pred = tf.clip_by_value(tf.nn.sigmoid(y_pred),0.0001,0.9999) #일정 epochs 이후 loss가 nan으로 발생하는 문제 발견
        return -tf.reduce_mean(
            tf.reduce_sum(
                tf.multiply(y_true, tf.math.log(y_pred)) + tf.multiply(1-y_true, tf.math.log(1-y_pred)),1))
    def _race_loss(self, y_true, y_pred): # sparse cross entropy loss
        # (7822, 2) 0 인덱스는 성별 1인덱스는 인종
        y_true = tf.split(y_true,2,1) # 분리!
        y_true_0 = tf.cast(y_true[1], tf.int32)  # (7822 , 1)
        y_true = tf.one_hot(y_true_0, depth=5, dtype=tf.float32)  # (7822, 1, 5) / 라벨 행렬을 만듬
        #printt(y_true.shape())
        #print(y_true)
        y_true = tf.squeeze(y_true, 1) # 1 삭제 (7822, 5)

        y_pred = tf.slice(y_pred, [0, 1], [-1, 5])
        y_pred = tf.clip_by_value(tf.nn.softmax(y_pred, 1), 0.0001, 1)
        return -tf.reduce_mean(
            tf.reduce_sum(
                tf.multiply(y_true, tf.math.log(y_pred)), 1))

    def _build_layers(self):
        layers = [
            tf.keras.layers.Conv2D(32, (3,3), padding='SAME', input_shape=(200,200,3),activation='relu'),
            tf.keras.layers.Conv2D(32, (3, 3), padding='SAME',activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2), (2, 2), padding='SAME'),

            tf.keras.layers.Conv2D(64, (3, 3), padding='SAME',activation='relu'),
            tf.keras.layers.Conv2D(64, (3, 3), padding='SAME',activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2), (2, 2), padding='SAME'),

            tf.keras.layers.Conv2D(128, (3, 3), padding='SAME',activation='relu'),
            tf.keras.layers.Conv2D(128, (3, 3), padding='SAME',activation='relu'),
            tf.keras.layers.Conv2D(128, (3, 3), padding='SAME',activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2), (2, 2), padding='SAME'),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(300, activation='relu'),
            #tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(60, activation='relu'),
            tf.keras.layers.Dense(6)
        ]
        return layers

    def _gender_accuracy(self,y_true,y_pred):
        y_true = tf.split(y_true,2,1) # _my_loss랑 똑같은 텐서 분리 주석 참조
        y_true = tf.cast(y_true[0], tf.float32)
        #y_true = tf.one_hot(y_true_0, depth=1, dtype=tf.float32) 이진분류제외
        # printt(y_true.shape())
        # print(y_true)
       #y_true = tf.squeeze(y_true, 1)  # 1 삭제 이진분류라 제외
        y_pred = tf.split(y_pred, 6, 1)[0]

        return tf.keras.metrics.binary_accuracy(y_true,y_pred)

    def _race_accuracy(self,y_true,y_pred):

        y_pred = tf.slice(y_pred, [0, 1], [-1, 5])
        y_true = tf.split(y_true,2,1) # _my_loss랑 똑같은 텐서 분리 주석 참조
        y_true = tf.cast(y_true[1], tf.int32)
        y_true = tf.one_hot(y_true, depth=5, dtype=tf.float32)

        # printt(y_true.shape())
        # print(y_true)
        y_true = tf.squeeze(y_true, 1)  # 1 삭제

        return tf.reduce_mean(
            tf.cast(
                tf.equal(tf.argmax(y_true, 1), tf.argmax(y_pred, 1)), tf.float32))

    def recall(self, y_true, y_pred):
        y_true = tf.split(y_true, 2, 1)  # loss랑 똑같은 텐서 분리
        y_true = tf.cast(y_true[0], tf.float32)

        y_pred = tf.split(y_pred, 6, 1)[0]
        # clip(t, clip_value_min, clip_value_max) : clip_value_min~clip_value_max 이외 가장자리를 깎아 낸다
        # round : 반올림한다
        y_target_yn = K.round(K.clip(y_true, 0, 1))  # 실제값을 0(Negative) 또는 1(Positive)로 설정한다
        y_pred_yn = K.round(K.clip(y_pred, 0, 1))  # 예측값을 0(Negative) 또는 1(Positive)로 설정한다

        # True Positive는 실제 값과 예측 값이 모두 1(Positive)인 경우이다
        count_true_positive = K.sum(y_target_yn * y_pred_yn)

        # (True Positive + False Negative) = 실제 값이 1(Positive) 전체
        count_true_positive_false_negative = K.sum(y_target_yn)

        # Recall =  (True Positive) / (True Positive + False Negative)
        # K.epsilon()는 'divide by zero error' 예방차원에서 작은 수를 더한다
        recall = count_true_positive / (count_true_positive_false_negative + K.epsilon())

        # return a single tensor value
        return recall

    def f1score(self,y_true, y_pred):
        _recall = self.recall(y_true, y_pred)
        _precision = self.precision(y_true, y_pred)
        # K.epsilon()는 'divide by zero error' 예방차원에서 작은 수를 더한다
        _f1score = (2 * _recall * _precision) / (_recall + _precision + K.epsilon())

        # return a single tensor value
        return _f1score

    def precision(self, y_true, y_pred):
        y_true = tf.split(y_true, 2, 1)  # _~~_loss랑 똑같은 텐서 분리
        y_true = tf.cast(y_true[0], tf.float32)

        y_pred = tf.split(y_pred, 6, 1)[0]
        y_pred_yn = K.round(K.clip(y_pred, 0, 1))  # 예측값을 0(Negative) 또는 1(Positive)로 설정한다
        y_target_yn = K.round(K.clip(y_true, 0, 1))  # 실제값을 0(Negative) 또는 1(Positive)로 설정한다

        # True Positive는 실제 값과 예측 값이 모두 1(Positive)인 경우이다
        count_true_positive = K.sum(y_target_yn * y_pred_yn)

        # (True Positive + False Positive) = 예측 값이 1(Positive) 전체
        count_true_positive_false_positive = K.sum(y_pred_yn)

        # Precision = (True Positive) / (True Positive + False Positive)
        # K.epsilon()는 'divide by zero error' 예방차원에서 작은 수를 더한다
        precision = count_true_positive / (count_true_positive_false_positive + K.epsilon())

        # return a single tensor value
        return precision

    def fit(self, x, t, epochs, te_x, te_y):
        self.history = self.model.fit(x, t, epochs=epochs,validation_data=(te_x,te_y))

    def evaluate(self, x, t):
        return self.model.evaluate(x,t)

def main():
    #mnist = tf.keras.datasets.mnist
    #(tr_X, tr_t), (te_X, te_t) = mnist.load_data()
    #tr_X, te_X = (tr_X / 255.).reshape(-1, 28, 28, 1), (te_X / 255.).reshape(-1, 28, 28, 1)
    #Data = load_data()
    path = './image_data.npy'
    if os.path.isfile(path):
        X_train, X_test, Y_train, Y_test = np.load("./image_data.npy",allow_pickle=True)
    else:
        Data = load_data()
        X_train = Data.X_train
        X_test = Data.X_test
        Y_train = Data.Y_train
        Y_test = Data.Y_test

    X_test = (X_test.astype('float32') / 255.).reshape(-1, 200, 200, 3)
    #print(X_train.shape)


    X_train = np.vsplit(X_train,5)
    Y_train = np.vsplit(Y_train,5)
    cnn = Custom_CNN()
    print(cnn.model.summary())
    for z in range(5):
        print("#################  "+str(z)+"번째 데이터조합  ###########################")
        k_Fold_X_test = X_train[z]
        k_Fold_Y_test = Y_train[z]
        k_Fold_list = []
        for j in range(5):
            if z != j:
                k_Fold_list.append(j)
        k_Fold_X_train = np.concatenate((X_train[k_Fold_list[0]],X_train[k_Fold_list[1]],
                                         X_train[k_Fold_list[2]],X_train[k_Fold_list[3]]))
        k_Fold_Y_train = np.concatenate((Y_train[k_Fold_list[0]], Y_train[k_Fold_list[1]],
                                         Y_train[k_Fold_list[2]], Y_train[k_Fold_list[3]]))
        k_Fold_X_train = (k_Fold_X_train.astype('float32') / 255.).reshape(-1, 200, 200, 3)
        cnn = Custom_CNN()
        cnn.fit(k_Fold_X_train, k_Fold_Y_train, epochs=5, te_x=X_test, te_y=Y_test)
        his = cnn.history
        print(cnn.evaluate(X_test, Y_test))

        confusion = np.zeros((5, 5))
        prediction = cnn.model.predict(X_test)
        num_of_lables = np.zeros(5)
        for i in range(len(prediction)):
            temp = prediction[i][1:]
            index1 = np.argmax(temp)
            # print(prediction[i])
            # print(temp)
            _true = Y_test[i][1]
            num_of_lables[_true] = num_of_lables[_true] + 1
            confusion[_true][index1] = confusion[_true][index1] + 1
        print(confusion)

        plt.subplot(3,2,1)
        plt.plot(his.history['loss'])
        plt.plot(his.history['val_loss'])
        plt.legend(['loss', 'val_loss'], loc='upper left')

        plt.subplot(3, 2, 2)
        plt.plot(his.history['_gender_accuracy'])
        plt.plot(his.history['val__gender_accuracy'])
        plt.legend(['_gender_accuracy', 'val__gender_accuracy'], loc='upper left')

        plt.subplot(3, 2, 3)
        plt.plot(his.history['_race_accuracy'])
        plt.plot(his.history['val__race_accuracy'])
        plt.legend(['race_accuracy', 'val_race_accuracy'], loc='upper left')

        plt.subplot(3, 2, 4)
        plt.plot(his.history['precision'])
        plt.plot(his.history['val_precision'])
        plt.legend(['precision', 'val_precision'], loc='upper left')

        plt.subplot(3, 2, 5)
        plt.plot(his.history['recall'])
        plt.plot(his.history['val_recall'])
        plt.legend(['recall', 'val_recall'], loc='upper left')

        plt.subplot(3, 2, 6)
        plt.plot(his.history['f1score'])
        plt.plot(his.history['val_f1score'])
        plt.legend(['f1score', 'val_f1score'], loc='upper left')

        plt.savefig("data_"+str(z+1)+".jpg")
        plt.close()
    #X_train = np.concatenate((X_train[0],X_train[1],X_train[2],X_train[3]))
    #Y_train = np.concatenate((Y_train[0], Y_train[1], Y_train[2], Y_train[3]))
    #print(X_train.shape)


    #y_true = tf.slice(Y_train,[0,1],[-1,1])
    #print(y_true.shape)
    #y_true_0 = tf.cast(y_true[0], tf.int32)
    #print(y_true_0.shape)
    #y_true = tf.one_hot(y_true_0, depth=1, dtype=tf.float32)
    #print(y_true.shape)



    #confusion matrix
    confusion = np.zeros((5,5))
    prediction = cnn.model.predict(X_test)
    num_of_lables = np.zeros(5)
    for i in range(len(prediction)):
        temp = prediction[i][1:]
        index1 = np.argmax(temp)
        #print(prediction[i])
        #print(temp)
        _true = Y_test[i][1]
        num_of_lables[_true] = num_of_lables[_true]+1
        confusion[_true][index1] = confusion[_true][index1] + 1
    print(confusion)
    print(num_of_lables)

if __name__ == '__main__':
    main()