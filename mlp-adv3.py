import tensorflow as tf
import numpy as mp
import matplotlib.pyplot as plt

class MLP:
    def __init__(self):
        self.layers = self._build_layers()
        self.model = tf.keras.Sequential(self.layers)

        self.model.compile(optimizer=tf.keras.optimizers.SGD(1e-1),
                           loss=self._my_loss,
                           metrics=[self._my_accuracy]
                           )
    def _my_loss(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32) # None , 1
        y_true = tf.one_hot(y_true, depth=10, dtype=tf.float32)# None, 1, 10
        y_true = tf.squeeze(y_true,1)
        y_pred = tf.nn.softmax(y_pred, 1)

        #cross entropy -sum t * log y
        return -tf.reduce_mean(
            tf.reduce_sum(
                tf.multiply(y_true, tf.math.log(y_pred)), 1))

    def _my_accuracy(self,y_true,y_pred):
        y_true = tf.cast(y_true, tf.int32)  # None , 1
        y_true = tf.one_hot(y_true, depth=10, dtype=tf.float32)  # None, 1, 10
        y_true = tf.squeeze(y_true, 1)

        return tf.reduce_mean(
            tf.cast(
                tf.equal(tf.argmax(y_true, 1), tf.argmax(y_pred, 1)), tf.float32))
    def fit(self, x, t, epochs):
        self.model.fit(x, t, epochs=epochs)
    def _build_layers(self):
        layers=[
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(500, activation='sigmoid'),
            tf.keras.layers.Dense(100, activation='sigmoid'),
            tf.keras.layers.Dense(10)
        ]
        return layers
def main():
    mnist = tf.keras.datasets.mnist
    (tr_X, tr_t), (te_X, te_t) = mnist.load_data()
    tr_X, te_X = tr_X / 255., te_X / 255.

    model = MLP()
    model.fit(tr_X, tr_t, epochs=10)

if __name__ == '__main__':
    main()