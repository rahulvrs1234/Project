"""Transfer-learning based COVID-19 image classifier reconstructed from the project snippet."""

import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from imutils import paths
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D, Dense, Dropout, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical


def build_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", required=True, help="path to input dataset")
    parser.add_argument("-p", "--plot", type=str, default="plot.png", help="path to output plot")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="covid19.model",
        help="path to output trained model",
    )
    return parser


def main():
    args = vars(build_argument_parser().parse_args())

    init_lr = 1e-3
    epochs = 2
    batch_size = 8

    print("[INFO] loading images...")
    image_paths = list(paths.list_images(args["dataset"]))
    data = []
    labels = []

    for image_path in image_paths:
        label = image_path.split(os.path.sep)[-2]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        data.append(image)
        labels.append(label)

    data = np.array(data, dtype="float32") / 255.0
    labels = np.array(labels)

    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    labels = to_categorical(labels)

    trainX, testX, trainY, testY = train_test_split(
        data,
        labels,
        test_size=0.20,
        stratify=labels,
        random_state=42,
    )

    train_aug = ImageDataGenerator(rotation_range=15, fill_mode="nearest")

    base_model = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
    head_model = base_model.output
    head_model = AveragePooling2D(pool_size=(4, 4))(head_model)
    head_model = Flatten(name="flatten")(head_model)
    head_model = Dense(64, activation="relu")(head_model)
    head_model = Dropout(0.5)(head_model)
    head_model = Dense(2, activation="softmax")(head_model)
    model = Model(inputs=base_model.input, outputs=head_model)

    for layer in base_model.layers:
        layer.trainable = False

    print("[INFO] compiling model...")
    opt = Adam(learning_rate=init_lr, decay=init_lr / epochs)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

    print("[INFO] training head...")
    history = model.fit(
        train_aug.flow(trainX, trainY, batch_size=batch_size),
        steps_per_epoch=max(1, len(trainX) // batch_size),
        validation_data=(testX, testY),
        validation_steps=max(1, len(testX) // batch_size),
        epochs=epochs,
    )

    print("[INFO] evaluating network...")
    pred_idxs = model.predict(testX, batch_size=batch_size)
    pred_idxs = np.argmax(pred_idxs, axis=1)

    print(classification_report(testY.argmax(axis=1), pred_idxs, target_names=lb.classes_))

    cm = confusion_matrix(testY.argmax(axis=1), pred_idxs)
    total = cm.sum()
    acc = (cm[0, 0] + cm[1, 1]) / total
    sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])

    print(cm)
    print(f"acc: {acc:.4f}")
    print(f"sensitivity: {sensitivity:.4f}")
    print(f"specificity: {specificity:.4f}")

    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, epochs), history.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), history.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, epochs), history.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), history.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy on COVID-19 Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(args["plot"])

    print("[INFO] saving COVID-19 detector model...")
    model.save(args["model"], save_format="h5")


if __name__ == "__main__":
    main()
