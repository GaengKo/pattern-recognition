import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt



class Rnn:
    def __init__(self):
        self.model = tf.keras.Sequential(self._build_layers())
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def _build_layers(self):
        layers = [
            tf.keras.layers.LSTM(64, input_shape=(None, 28)),# None 은 Time step의 크기를 입력에 따라 유동적으로 하겠다는 의미
            tf.keras.layers.Dense(10, activation='softmax')
        ]
        return layers
    def fit(self, x, t, epochs):
        self.model.fit(x, t, epochs=epochs)

    def evaluate(self, x, t):
        return self.model.evaluate(x,t)

def main():
    mnist = tf.keras.datasets.mnist
    (tr_X, tr_t), (te_X, te_t) = mnist.load_data()
    tr_X, te_X = tr_X / 255., te_X / 255.

    rnn = Rnn()
    rnn.fit(tr_X,tr_t,3)

    print(rnn.evaluate(te_X, te_t))


if __name__ == '__main__':
    main()