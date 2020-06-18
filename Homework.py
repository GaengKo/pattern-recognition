import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from sklearn.model_selection import train_test_split
import random

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
        #self.model.compile(optimizer='adam',  loss=self._binary_loss , metrics=[self._binary_accuracy])
        self.model.compile(optimizer='adam',  loss=self._sparse_loss , metrics=[self._sparse_accuracy])
        self.model.compile(optimizer='adam',  loss=self._sum_loss , metrics=[self._binary_accuracy, self._sparse_accuracy])

    def _sum_loss(self, y_true, y_pred):
        return (self._binary_loss(y_true,y_pred)+self._sparse_loss(y_true,y_pred))/2.

    def _binary_loss(self, y_true, y_pred):
        # (7822, 2) 0 인덱스는 성별 1인덱스는 인종
        y_true = tf.split(y_true, 2, 1)  # 분리!
        y_true = tf.cast(y_true[0], tf.float32)  # (7822 , 1)
        y_pred = tf.split(y_pred,6,1)[0]

        y_pred = tf.clip_by_value(tf.nn.sigmoid(y_pred),0.001,0.999) #일정 epochs 이후 loss가 nan으로 발생하는 문제 발견
        #return tf.keras.losses.binary_crossentropy(y_true,y_pred)
        return -tf.reduce_mean(
            tf.reduce_sum(
                tf.multiply(y_true, tf.math.log(y_pred)) + tf.multiply(1-y_true, tf.math.log(1-y_pred)),1))
    def _sparse_loss(self, y_true, y_pred): # sparse cross entropy loss
        # (7822, 2) 0 인덱스는 성별 1인덱스는 인종
        y_true = tf.split(y_true,2,1) # 분리!
        y_true_0 = tf.cast(y_true[1], tf.int32)  # (7822 , 1)
        y_true = tf.one_hot(y_true_0, depth=5, dtype=tf.float32)  # (7822, 1, 5) / 라벨 행렬을 만듬
        #printt(y_true.shape())
        #print(y_true)
        y_true = tf.squeeze(y_true, 1) # 1 삭제 (7822, 5)

        y_pred = tf.slice(y_pred, [0, 1], [-1, 5])
        y_pred = tf.nn.softmax(y_pred, 1)

        return -tf.reduce_mean(
            tf.reduce_sum(
                tf.multiply(y_true, tf.math.log(y_pred)), 1))

    def _build_layers(self):
        layers = [
            tf.keras.layers.Conv2D(32, (3,3), padding='SAME', activation='relu', input_shape=(200,200,3)),
            tf.keras.layers.MaxPooling2D((2,2),(2,2),padding='SAME'),
            tf.keras.layers.Conv2D(32, (3, 3), padding='SAME', activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2), (2, 2), padding='SAME'),
            tf.keras.layers.Conv2D(64, (3, 3), padding='SAME', activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2), (2, 2), padding='SAME'),
            tf.keras.layers.Conv2D(64, (3, 3), padding='SAME', activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2), (2, 2), padding='SAME'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(30, activation='relu'),
            tf.keras.layers.Dense(6)
        ]
        return layers

    def _binary_accuracy(self,y_true,y_pred):
        y_true = tf.split(y_true,2,1) # _my_loss랑 똑같은 텐서 분리 주석 참조
        y_true = tf.cast(y_true[0], tf.float32)
        #y_true = tf.one_hot(y_true_0, depth=1, dtype=tf.float32)
        # printt(y_true.shape())
        # print(y_true)
       #y_true = tf.squeeze(y_true, 1)  # 1 삭제
        y_pred = tf.split(y_pred, 6, 1)[0]

        return tf.keras.metrics.binary_accuracy(y_true,y_pred)
        #return tf.reduce_mean(
         #   tf.cast(
         #       tf.equal(y_true, y_pred), tf.float32))
    def _sparse_accuracy(self,y_true,y_pred):
        y_true = tf.split(y_true,2,1) # _my_loss랑 똑같은 텐서 분리 주석 참조
        y_true_0 = tf.cast(y_true[1], tf.int32)
        y_true = tf.one_hot(y_true_0, depth=5, dtype=tf.float32)
        # printt(y_true.shape())
        # print(y_true)
        y_true = tf.squeeze(y_true, 1)  # 1 삭제
        y_pred = tf.slice(y_pred, [0,1], [-1,5])
        return tf.reduce_mean(
            tf.cast(
                tf.equal(tf.argmax(y_true, 1), tf.argmax(y_pred, 1)), tf.float32))
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

    #y_true = tf.slice(Y_train,[0,1],[-1,1])
    #print(y_true.shape)
    #y_true_0 = tf.cast(y_true[0], tf.int32)
    #print(y_true_0.shape)
    #y_true = tf.one_hot(y_true_0, depth=1, dtype=tf.float32)
    #print(y_true.shape)
    cnn = Custom_CNN()
    print(cnn.model.summary())
    X_train = (X_train.astype('float32') / 255.).reshape(-1, 200, 200, 3)
    X_test = (X_test.astype('float32') / 255.).reshape(-1, 200, 200, 3)

    cnn.fit(X_train, Y_train, epochs=10,te_x=X_test,te_y=Y_test)
    print(cnn.evaluate(X_test, Y_test))
if __name__ == '__main__':
    main()