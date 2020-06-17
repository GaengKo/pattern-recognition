import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mimg
from sk

def scale(a):
    mx = np.max(a, (0, 1)).reshape((1,1,3))
    mn = np.min(a, (0, 1)).reshape((1, 1, 3))

    return (a-mn)/(mx-mn)

def plot_weights(weights):
    fig, ax = plt.subplots(8, 8)
    for  h in range(8):
        for w in range(8):
            ax[h, w].imshow(scale(weights[:,:,:,h*8+w]))
    fig.show()

def plot_featuremap(data, layers):
    z = data
    for layer in layers:
        z = layer(z)
        if len(z.shape) > 2:
            fig = plt.figure()
            plt.imshow(z[0,:,:,0], cmap='gray')
    

def main():
    vgg16 = tf.keras.applications.VGG16()
    vgg16.summary()
    layer1_w = vgg16.trainable_variables[0]

    plot_weights(layer1_w)

    plt.show()
if __name__ == '__main__':
    main()
