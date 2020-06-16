import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
print(tf.__version__)

class MLP(tf.keras.Model):
    def __init__(self):
        super(MLP,self).__init__()
        self.mlp_layers = self._build_layers()
        self.loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        self.opt = tf.keras.optimizers.SGD(1e-1)

    def _build_layers(self):
        layers = [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(500, activation='sigmoid'),
            tf.keras.layers.Dense(100, activation='sigmoid'),
            tf.keras.layers.Dense(10)
        ]
        return layers
    @tf.function
    def call(self,x, training=False):
        if not tf.is_tensor(x):
            x = tf.convert_to_tensor(x)

        z = x
        for l in self.mlp_layers:
            z = l(z)

        return z
    @tf.function
    def fit(self, x, t):
        if not tf.is_tensor(t):
            t = tf.convert_to_tensor(t, dtype=tf.int32)
        t = tf.one_hot(t, depth=10, dtype=tf.float32)

        with tf.GradientTape() as g:
            y = self(x, training=True)
            var_list = self.trainable_variables

            loss = self.loss_func(t, y)
            grad = g.gradient(loss, var_list)
            self.opt.apply_gradients(zip(grad, var_list))

            return loss

    def accuracy(self,x, t):
        if not tf.is_tensor(t):
            t = tf.convert_to_tensor(t, dtype=tf.int32)

        t = tf.one_hot(t, depth=10, dtype=tf.float32)

        y = self(x)
        return tf.reduce_mean(
            tf.cast(
                tf.equal(
                    tf. argmax(y, 1), tf.argmax(t,1)
                ), tf.float32
            )
        )


def main():
    mnist = tf.keras.datasets.mnist
    (tr_X, tr_t), (te_X, te_t) = mnist.load_data()
    tr_X, te_X = tr_X / 255., te_X / 255.

    n_batch = 100
    n_epoch = 2
    mlp = MLP()
    losses = []
    tr_accuracies = []
    te_accuracies = []
    fig, ax = plt.subplots(1, 2)
    for _ in range(n_epoch):
        for i in range(0,len(tr_X), n_batch):
            tr_X_mb, tr_t_mb = tr_X[i:i+n_batch], tr_t[i:i+n_batch]
            loss = mlp.fit(tr_X_mb, tr_t_mb).numpy()
            if i % 1000 == 0:
                losses.append(loss)
                tr_acc = mlp.accuracy(tr_X_mb,tr_t_mb).numpy()
                te_acc = mlp.accuracy(te_X,te_t).numpy()
                tr_accuracies.append(tr_acc)
                te_accuracies.append(te_acc)
                ax[0].cla()
                ax[1].cla()

                ax[0].plot(losses)
                ax[1].plot(tr_accuracies, c= 'b', label='training accuracy')
                ax[1].plot(te_accuracies, c='r', label='test accuracy')
                plt.legend()
                plt.tight_layout()
                plt.draw()
                plt.pause(0.1)

    plt.show()
if __name__=='__main__':
    main()