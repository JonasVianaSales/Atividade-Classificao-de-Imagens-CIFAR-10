import numpy as np
import matplotlib.pyplot as plt
from keras import datasets, models
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical


def model():
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    model = Sequential([
                    Conv2D(32, (2,2), activation="relu", input_shape=(32, 32, 3)),
                    MaxPooling2D(2),
                    Conv2D(32, (2,2), activation="relu"),
                    MaxPooling2D(2),
                    Flatten(),
                    Dense(32, activation="relu"),
                    Dense(10, activation="softmax")
    ])
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]    
    )

    result = model.fit(x_train, y_train, batch_size=128, epochs=16, verbose=2, validation_data=(x_test, y_test))

    model.save('C:/Users/jonas/Desktop/Backup 13.02.2023/Faculdade/Módulo 10/Semana 7/Classificação de Imagens (CIFAR-10)/model.h5')

def teste():
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    plt.imshow(x_train[0])

#teste()
model()