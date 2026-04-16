"""Custom CNN builder reconstructed from the available project snippet."""

import os
import pickle
import time

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential

# Use the line below only when you do not have an NVIDIA graphic card and CUDA support.
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

NAME = "covid19_and_normal"
PATH = os.path.join("logs", NAME)

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.5)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

with open("X.pickle", "rb") as pickle_in_x:
    X = pickle.load(pickle_in_x)
with open("Y.pickle", "rb") as pickle_in_y:
    Y = pickle.load(pickle_in_y)

X = X / 255.0

conv_layers = [3]
conv_sizes = [64]
dense_layers = [1]

for conv_layer in conv_layers:
    for conv_size in conv_sizes:
        for dense_layer in dense_layers:
            name = f"{conv_layer}-conv_layer-{conv_size}-conv_size-{dense_layer}-dense_layer-{int(time.time())}"
            path = os.path.join("logs", name)
            tensorboard = TensorBoard(log_dir=path, profile_batch=0)

            model = Sequential()
            model.add(Conv2D(conv_size, (3, 3), input_shape=X.shape[1:]))
            model.add(Activation("relu"))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            for _ in range(conv_layer - 1):
                model.add(Conv2D(conv_size, (3, 3)))
                model.add(Activation("relu"))
                model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Dropout(0.3))
            model.add(Flatten())

            for _ in range(dense_layer):
                model.add(Dense(conv_size))
                model.add(Activation("relu"))
                model.add(Dropout(0.5))

            model.add(Dense(1))
            model.add(Activation("sigmoid"))

            model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
            model.fit(X, Y, batch_size=9, epochs=14, validation_split=0.3, callbacks=[tensorboard])
            model.save("covid19_pneumonia_detection_cnn.model")
