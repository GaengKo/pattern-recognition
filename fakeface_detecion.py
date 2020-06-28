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
        pass

    def load_data(self, _type):

        self.X = []
        self.Y = []
        if _type == 'training':
            self.path1 = '../problem1_dataset(1)/training/fake/'
            self.path2 = '../problem1_dataset(1)/training/real/'
        else:
            self.path1 = '../problem1_dataset(1)/test/fake/'
            self.path2 = '../problem1_dataset(1)/test/real/'
        self.image_list1 = os.listdir(self.path1)
        self.image_list2 = os.listdir(self.path2)

        for i in self.image_list1:
            print(i)

        # suffle
        # random.shuffle(self.image_list)
        for i in range(len(self.image_list2)):
            im = Image.open(self.path2 + self.image_list2[i])
            im = im.convert("RGB")
            data = np.asarray(im)
            try:
                # self.Y.append(int(self.image_list[i].split('_')[2]))
                P_label = []

                P_label.append(0)
                P_label.append(0)
                P_label.append(0)
                P_label.append(0)

                self.Y.append(P_label)
                self.X.append(data)

            except Exception as e:
                print(e)

        print("################################################")
        for i in range(len(self.image_list1)):
            im = Image.open(self.path1 + self.image_list1[i])
            im = im.convert("RGB")
            data = np.asarray(im)
            try:
                # self.Y.append(int(self.image_list[i].split('_')[2]))
                P_label = []
                temp = self.image_list1[i].split('_')
                print(temp[2])
                P_label.append(int(temp[2][0]))
                P_label.append(int(temp[2][1]))
                P_label.append(int(temp[2][2]))
                P_label.append(int(temp[2][3]))
                self.Y.append(P_label)
                self.X.append(data)

            except Exception as e:
                print(e)

        return self.X, self.Y

    def make_npy(self):
        X_train, Y_train = self.load_data('training')
        X_test, Y_test = self.load_data('test')
        tmp = [[x,y] for x, y in zip(X_train, Y_train)]
        random.shuffle(tmp)
        X_train = [n[0] for n in tmp]
        Y_train = [n[1] for n in tmp]

        tmp = [[x, y] for x, y in zip(X_test, Y_test)]
        random.shuffle(tmp)
        X_test = [n[0] for n in tmp]
        Y_test = [n[1] for n in tmp]

        X_train = np.array(X_train)
        Y_train = np.array(Y_train)

        X_test = np.array(X_test)
        Y_test = np.array(Y_test)
        xy = (X_train, X_test, Y_train, Y_test)
        np.save("./face_data.npy", xy)




class Custom_CNN:
    def __init__(self):
        self.model = tf.keras.Sequential(self._build_layers())
       # sgd = tf.keras.optimizers.SGD(lr=0.001,decay=1e-6, momentum=0.5)
        adam = tf.keras.optimizers.Adam(lr=0.00001)
        self.model.compile(optimizer=adam,  loss=self._my_loss, metrics=[self._average_acc, self._left_eye_acc, self._right_eye_acc, self._nose_acc, self._mouth_acc])

    def _my_loss(self, y_true, y_pred):
        result = 0
        y_true = tf.split(y_true, 4, 1)  # 분리
        y_pred = tf.split(y_pred, 4, 1)
        for i in range(4):
            y_temp = tf.cast(y_true[i], tf.float32)  # (7822 , 1)
            y_temp_p = tf.clip_by_value( y_pred[i], 0.0001, 0.9999)
            result = result + tf.keras.losses.binary_crossentropy(y_temp,y_temp_p)
        result = result/4.
        return result



    def _build_layers(self):
        layers = [
            tf.keras.layers.Conv2D(32, (3,3), padding='SAME', input_shape=(600,600,3)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Conv2D(32, (3, 3), padding='SAME'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.MaxPooling2D((2, 2), (2, 2), padding='SAME'),

            tf.keras.layers.Conv2D(64, (3, 3), padding='SAME'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Conv2D(64, (3, 3), padding='SAME'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.MaxPooling2D((2, 2), (2, 2), padding='SAME'),

            tf.keras.layers.Conv2D(128, (3, 3), padding='SAME'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Conv2D(128, (3, 3), padding='SAME'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Conv2D(128, (3, 3), padding='SAME'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.MaxPooling2D((2, 2), (2, 2), padding='SAME'),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(60, activation='relu'),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(4,activation='sigmoid')
        ]
        return layers

    def _average_acc(self, y_true, y_pred):

        total_acc = self._left_eye_acc(y_true, y_pred)
        total_acc = total_acc + self._right_eye_acc(y_true, y_pred)
        total_acc = total_acc + self._nose_acc(y_true, y_pred)
        total_acc = total_acc + self._mouth_acc(y_true, y_pred)
        return total_acc/4.

    def _left_eye_acc(self,y_true,y_pred):

        y_true = tf.split(y_true, 4, 1)
        y_pred = tf.split(y_pred, 4, 1)
        y_true = tf.cast(y_true, tf.float32)

        return tf.keras.metrics.binary_accuracy(y_true[0],y_pred[0])
    def _right_eye_acc(self, y_true, y_pred):
        y_true = tf.split(y_true, 4, 1)
        y_pred = tf.split(y_pred, 4, 1)
        y_true = tf.cast(y_true, tf.float32)

        return tf.keras.metrics.binary_accuracy(y_true[1],y_pred[1])
    def _nose_acc(self, y_true, y_pred):
        y_true = tf.split(y_true, 4, 1)
        y_pred = tf.split(y_pred, 4, 1)
        y_true = tf.cast(y_true, tf.float32)

        return tf.keras.metrics.binary_accuracy(y_true[2],y_pred[2])
    def _mouth_acc(self, y_true, y_pred):
        y_true = tf.split(y_true, 4, 1)
        y_pred = tf.split(y_pred, 4, 1)
        y_true = tf.cast(y_true, tf.float32)

        return tf.keras.metrics.binary_accuracy(y_true[3],y_pred[3])


    def fit(self, x, t, epochs, te_x, te_y):
        checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='./best.model', monitor='val__average_acc', verbose=1, save_best_only=True)
        self.history = self.model.fit(x, t, epochs=epochs,validation_data=(te_x,te_y),callbacks=[checkpoint])

    def evaluate(self, x, t):
        return self.model.evaluate(x,t)

def main():
    path = './face_data.npy'
    if os.path.isfile(path):
        X_train, X_test, Y_train, Y_test = np.load("./face_data.npy", allow_pickle=True)
    else:
        Data = load_data()
        Data.make_npy()
        X_train, X_test, Y_train, Y_test = np.load("./face_data.npy", allow_pickle=True)

    X_train = (X_train.astype('float32') / 255.).reshape(-1, 600, 600, 3)
    X_test = (X_test.astype('float32') / 255.).reshape(-1, 600, 600, 3)
    #print(Y_train) #1428,4
    cnn = Custom_CNN()
    print(cnn.model.summary())

    cnn.fit(X_train,Y_train,100,X_test,Y_test)
    his = cnn.history
    plt.figure(figsize=(20,10))
    plt.subplot(1,2,1)
    plt.plot(his.history['loss'])
    plt.plot(his.history['val_loss'])
    plt.plot(his.history['_average_acc'])
    plt.plot(his.history['val__average_acc'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(
        ['loss', 'val_loss', 'left_eye_acc', 'total_acc', 'val_total_acc'], loc='upper left')


    plt.subplot(1, 2, 2)
    plt.plot(his.history['_left_eye_acc'])
    plt.plot(his.history['val__left_eye_acc'])

    plt.plot(his.history['_right_eye_acc'])
    plt.plot(his.history['val__right_eye_acc'])

    plt.plot(his.history['_nose_acc'])
    plt.plot(his.history['val__nose_acc'])

    plt.plot(his.history['_mouth_acc'])
    plt.plot(his.history['val__mouth_acc'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')


    plt.legend([ 'val_left_eye_acc', 'right_eye_acc', 'val_right_eye_acc', 'nose_acc', 'val_nose_acc'
                ,'mouth_acc','var_mouth_acc'], loc='upper left')
    plt.show()
    plt.savefig("fake_face.jpg")


if __name__ == '__main__':
    main()