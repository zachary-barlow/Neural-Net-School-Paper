
import tensorflow as tf

from tensorflow.keras.datasets import mnist, cifar10, cifar100

from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Activation

from tensorflow.keras import datasets, layers, models

import numpy as np
import random

import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

img_rows, img_cols, channels = 32, 32, 3
num_classes = 10

x_train = x_train / 255
x_test = x_test / 255

x_train = x_train.reshape((-1, img_rows, img_cols, channels))
x_test = x_test.reshape((-1, img_rows, img_cols, channels))


y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

def create_model():
    # Create the Convolutional base
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_rows, img_cols, channels)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    # Compile and train model
    model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
    return model


def adversarial_pattern(image, label):
    image = tf.cast(image, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = model(image)
        loss = tf.keras.losses.categorical_crossentropy(label, prediction, from_logits=False)

    gradient = tape.gradient(loss, image)

    signed_grad = tf.sign(gradient)

    return signed_grad

def generate_adversarials(batch_size, epsilon):
    while True:
        x = []
        y = []
        for N in range(batch_size):

            label = y_train[N]
            image = x_train[N]

            perturbations = adversarial_pattern(image.reshape((1, img_rows, img_cols, channels)), label).numpy()


            adversarial = image + perturbations * epsilon

            x.append(adversarial)
            y.append(y_train[N])
            if (N % 100 == 0): print(N/100)

        x = np.asarray(x).reshape((batch_size, img_rows, img_cols, channels))
        y = np.asarray(y)

        yield x, y

model = create_model()
n_train = 100
adv_batch_size = 100
epochs = 100
ep = 0.05


x_train, y_train = next(generate_adversarials(adv_batch_size, ep))

#import pickle
# with open('adv_train-cifar10.pickle', 'wb') as handle:
#     pickle.dump((x_train, y_train), handle, protocol=pickle.HIGHEST_PROTOCOL)

# exit()

#(x_train, y_train) = pickle.load(open("adv_train-cifar10.pickle", "rb" ))


history = model.fit(x_train[:n_train], y_train[:n_train], epochs=epochs, validation_data=(x_test, y_test))
print(history.history)


# Evaluate the model
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
print(test_acc)