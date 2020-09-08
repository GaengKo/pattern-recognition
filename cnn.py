import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

class Cnn:
    def __init__(self):
        self.model = tf.keras.Sequential(self._build_layers())
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def _build_layers(self):
        layers = [
            tf.keras.layers.Conv2D(8, (3,3), padding='SAME', activation='relu', input_shape=(28,28,1)),
            tf.keras.layers.MaxPooling2D((2,2),(2,2),padding='SAME'),
            tf.keras.layers.Conv2D(16, (3, 3), padding='SAME', activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2), (2, 2), padding='SAME'),
            tf.keras.layers.Conv2D(32, (3, 3), padding='SAME', activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2), (2, 2), padding='SAME'),
            tf.keras.layers.Conv2D(64, (3, 3), padding='SAME', activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2), (2, 2), padding='SAME'),
            tf.keras.layers.Conv2D(128, (3, 3), padding='SAME', activation='relu'),
            #f.keras.layers.MaxPooling2D((2, 2), (2, 2), padding='SAME'),
            tf.keras.layers.GlobalAveragePooling2D(),
            #tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10,activation='softmax')
        ]
        return layers
    def fit(self, x, t, epochs):
        self.model.fit(x, t, epochs=epochs)

    def evaluate(self, x, t):
        return self.model.evaluate(x,t)

def generate_cam(img_tensor, model, class_index, activation_layer):
    """
       params:
       -------
       img_tensor: resnet50 모델의 이미지 전처리를 통한 image tensor
       model: pretrained resnet50 모델 (include_top=True)
       class_index: 이미지넷 정답 레이블
       activation_layer: 시각화하려는 레이어 이름

       return:
       cam: cam 히트맵
       """
    inp = model.input
    A_k = model.get_layer(activation_layer).output
    outp = model.layers[-1].output

    ## 이미지 텐서를 입력해서
    ## 해당 액티베이션 레이어의 아웃풋(a_k)과
    ## 소프트맥스 함수 인풋의 a_k에 대한 gradient를 구한다.
    get_output = K.function([inp], [A_k, outp])
    [conv_output, predictions] = get_output([img_tensor])

    ## 배치 사이즈가 1이므로 배치 차원을 없앤다.
    conv_output = conv_output[0]

    ## 마지막 소프트맥스 레이어의 웨이트 매트릭스에서
    ## 지정한 레이블에 해당하는 횡벡터만 가져온다.
    weights = model.layers[-1].get_weights()[0][:, class_index]

    ## 추출한 conv_output에 weight를 곱하고 합하여 cam을 얻는다.
    cam = np.zeros(dtype=np.float32, shape=conv_output.shape[0:2])
    for k, w in enumerate(weights):
        cam += w * conv_output[:, :, k]

    cam = cv2.resize(cam, (224, 224))
    cam = cam / cam.max()

    return cam

def main():
    mnist = tf.keras.datasets.mnist
    (tr_X, tr_t), (te_X, te_t) = mnist.load_data()
    print(te_X.shape)
    tr_X, te_X = (tr_X / 255.).reshape(-1, 28, 28, 1), (te_X / 255.).reshape(-1, 28, 28, 1)

    cnn = Cnn()
    cnn.model.summary()
    cnn.fit(tr_X, tr_t, epochs=1)
    print(cnn.evaluate(te_X, te_t))

    get_output = tf.keras.backend.function([cnn.model.layers[0].input],
                                           [cnn.model.layers[-3].output, cnn.model.layers[-1].output])
    [conv_outputs, predictions] = get_output(te_X[:10])
    class_weights = cnn.model.layers[-1].get_weights()[0]

    output = []
    for num, idx in enumerate(np.argmax(predictions, axis=1)):
        cam = tf.matmul(np.expand_dims(class_weights[:, idx], axis=0),
                        np.transpose(np.reshape(conv_outputs[num], (2*2, 128))))
        cam = tf.keras.backend.eval(cam)
        output.append(cam)
    cam = np.reshape(cam, (2, 2))  # 2차원으로 변형
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # 0~1 사이로 정규화
    cam = np.expand_dims(np.uint8(255 * cam), axis=2)  # 0 ~ 255 사이로 정규화 및 차원 추가
    cam = cv2.applyColorMap(cv2.resize(cam, (28, 28)), cv2.COLORMAP_JET)
    # 컬러맵 처리 및 원본 이미지 크기와 맞춤
    cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)  # RGB로 바꿈
    (tr_X, tr_t), (te_X, te_t) = mnist.load_data()
    image = np.stack((te_X[0],)*3, axis=-1)
    print(image.shape)
    z = 0
    result = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # RGB로 바꿈
    result = result/2+cam[z]/2
    im = Image.fromarray(result)
    im.save('result.png')


if __name__ == '__main__':
    main()