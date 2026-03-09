import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
import numpy as np

IMG_SIZE = (128,128)
BATCH_SIZE = 32

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    "dataset/train",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

val_gen = val_datagen.flow_from_directory(
    "dataset/validation",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

# Custom CNN
def create_custom_cnn():

    model = models.Sequential()

    model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(128,128,3)))
    model.add(layers.MaxPooling2D(2,2))

    model.add(layers.Conv2D(64,(3,3),activation='relu'))
    model.add(layers.MaxPooling2D(2,2))

    model.add(layers.Flatten())

    model.add(layers.Dense(128,activation='relu'))
    model.add(layers.Dense(1,activation='sigmoid'))

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


# MobileNetV2
def create_mobilenet():

    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(128,128,3)
    )

    base_model.trainable = False

    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(256,activation='relu')(x)
    output = layers.Dense(1,activation='sigmoid')(x)

    model = tf.keras.Model(base_model.input, output)

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


# EfficientNetB0
def create_efficientnet():

    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(128,128,3)
    )

    base_model.trainable = False

    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(256,activation='relu')(x)
    output = layers.Dense(1,activation='sigmoid')(x)

    model = tf.keras.Model(base_model.input, output)

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


# Train models
models_dict = {
    "CustomCNN": create_custom_cnn(),
    "MobileNetV2": create_mobilenet(),
    "EfficientNetB0": create_efficientnet()
}

results = {}

for name, model in models_dict.items():

    print("\nTraining:", name)

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=10
    )

    val_acc = max(history.history['val_accuracy'])

    results[name] = val_acc

    model.save(f"{name}_model.h5")


# Find best model
best_model = max(results, key=results.get)

print("\nBest Model:", best_model)
print("Validation Accuracy:", results[best_model])