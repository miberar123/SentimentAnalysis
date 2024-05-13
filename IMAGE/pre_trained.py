import os

import numpy as np
import tensorflow as tf
from keras import layers, models
from keras.applications import VGG16


def main():

    base_dir = "IMAGE/data"
    train_dir = os.path.join(base_dir, "train")
    valid_dir = os.path.join(base_dir, "test")

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir, target_size=(150, 150), batch_size=32, class_mode="categorical"
    )

    validation_generator = test_datagen.flow_from_directory(
        valid_dir, target_size=(150, 150), batch_size=32, class_mode="categorical"
    )

    conv_base = VGG16(weights="imagenet", include_top=False, input_shape=(150, 150, 3))

    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(train_generator.num_classes, activation="softmax"))

    conv_base.trainable = False

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    history = model.fit(
        train_generator,
        steps_per_epoch=50,
        epochs=5,
        validation_data=validation_generator,
        validation_steps=50,
    )

    model.save("vgg16-5epoch.h5")

    test_loss, test_acc = model.evaluate(validation_generator, steps=50)
    print(f"Model Accuracy: {test_acc*100:.2f}%")


if __name__ == "__main__":
    main()
