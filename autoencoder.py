import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class Autoencoder:
    def __init__(self):
        self.encoder = tf.keras.Sequential(self._build_encoder_layer())
        self.decoder = tf.keras.Sequential(self._build_decoder_layer())
        self.autoencoder = tf.keras.Sequential([
            self.encoder,
            self.decoder
            ])
        self.autoencoder.compile(optimizer='Adam', loss='binary_crossentropy')


    def _build_encoder_layer(self):
        layers = [
          tf.keras.layers.Flatten(input_shape=(28,28)),
            tf.keras.layers.Dense(300, activation='relu'),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(50, activation='relu')

        ]
        return layers
    def _build_decoder_layer(self):
        layers = [
            tf.keras.layers.Dense(100, activation='relu', input_shape=(50,)),
            tf.keras.layers.Dense(300, activation='relu'),
            tf.keras.layers.Dense(784, activation='sigmoid'),
            tf.keras.layers.Reshape(target_shape=(28,28))
        ]
        return layers

    def fit(self, x, epochs):
        self.autoencoder.fit(x, x, epochs=epochs)
    def reconstruct(self, x):
        return self.autoencoder(x)

    def encoder(self, x):
        return self.encoder(x)

class MLP:
    def __init__(self):
        self.model = tf.keras.Sequential(self._build_layers())
        self.model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def _build_layers(self):
        layers = [
            tf.keras.layers.Dense(100,activation='relu', input_shape=(50,)),
            tf.keras.layers.Dense(10, activation='softmax')
        ]
        return layers

    def fit(self, x, t, epochs):
        self.model.fit(x, t, epochs=epochs)

    def evaluate(self, x, t):
        return self.model.evaluate(x, t)
def main():
    mnist = tf.keras.datasets.mnist
    (tr_X, tr_t), (te_X, te_t) = mnist.load_data()
    tr_X, te_X = tr_X / 255., te_X / 255.

    ae = Autoencoder()
    ae.fit(tr_X, epochs=3)

    te_X_r = ae.reconstruct(te_X[:3])
    for i in range(3):
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(te_X[i], cmap='gray'),
        ax[1].imshow(te_X_r[i], cmap='gray')
        fig.show()


    cls = MLP()
    cls.fit(ae.encoder(tr_X), tr_t, epochs=3)

    print(cls.evaluate(ae.encoder(te_X), te_t))

    plt.show()
if __name__ == '__main__':
    main()