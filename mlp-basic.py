import tensorflow as tf


mnist = tf.keras.datasets.mnist
(tr_X, tr_t), (te_X, te_t) = mnist.load_data()
tr_X, te_X = tr_X / 255., te_X/255.
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(500,activation='sigmoid'),
    tf.keras.layers.Dense(100, activation='sigmoid'),
    tf.keras.layers.Dense(10, activation='softmax')

])
#mnist dataset이 인덱스형석 라벨인경우 sparse 를 씀
model.compile(optimizer='sgd',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
#모델 형태 출력
model.summary()


model.fit(tr_X,tr_t,epochs=4)
accuracy = model.evaluate(te_X,te_t)
print(accuracy)