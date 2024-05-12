import os

import tensorflow as tf
from tqdm import tqdm

base_dir = "IMAGE/data"
train_dir = os.path.join(base_dir, "train")
validation_dir = os.path.join(base_dir, "validation")
test_dir = os.path.join(base_dir, "test")


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
    validation_dir, target_size=(150, 150), batch_size=32, class_mode="categorical"
)


def save_augmented_images_by_class(generator, base_save_dir, num_images=1000):

    class_indices = generator.class_indices
    class_names = list(class_indices.keys())

    os.makedirs(base_save_dir, exist_ok=True)

    for class_name in class_names:
        os.makedirs(os.path.join(base_save_dir, class_name), exist_ok=True)

    for i, (imgs, labels) in tqdm(
        enumerate(generator), total=num_images, desc="Saving images"
    ):
        if i >= num_images:
            break
    for img, label in zip(imgs, labels):
        class_name = class_names[label.argmax()]
        save_path = os.path.join(base_save_dir, class_name, f"augmented_{i}.png")
        tf.keras.utils.save_img(save_path, img)


augmented_train_save_dir = os.path.join(base_dir, "augmented/train")
save_augmented_images_by_class(train_generator, augmented_train_save_dir)

"""
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(150, 150, 3)
        ),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(len(train_generator.class_indices), activation="softmax"),
    ]
)

# Compile the model
model.compile(
    loss="categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["accuracy"],
)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=100,  # Adjust based on the size of your dataset
    epochs=30,  # Adjust based on the desired number of epochs
    validation_data=validation_generator,
    validation_steps=50,  # Adjust based on the size of your validation set
)
"""
