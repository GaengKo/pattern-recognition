import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class Cnn:
    def __init__(self):
        self.model = tf.keras.Sequential(self._build_layers())
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def _build_layers(self):
        layers = [
            tf.keras.layers.Conv2D(8, (3,3), padding='SAME', activation='relu', input_shape=(28,28,1)),
            tf.keras.layers.MaxPooling2D((2,2),(2,2),padding='SAME'),
            tf.keras.layers.Conv2D(3, (3, 3), padding='SAME', activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2), (2, 2), padding='SAME'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10,activation='softmax')
        ]
        return layers
    def fit(self, x, t, epochs):
        self.model.fit(x, t, epochs=epochs)

    def evaluate(self, x, t):
        return self.model.evaluate(x,t)

def main():
    mnist = tf.keras.datasets.mnist
    (tr_X, tr_t), (te_X, te_t) = mnist.load_data()
    tr_X, te_X = (tr_X / 255.).reshape(-1, 28, 28, 1), (te_X / 255.).reshape(-1, 28, 28, 1)

    cnn = Cnn()
    cnn.fit(tr_X, tr_t, epochs=3)
    print(cnn.evaluate(te_X, te_t))

if __name__ == '__main__':
    main()