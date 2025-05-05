import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import logging
import time


os.environ['CUDA_VISIBLE_DEVICES'] = "0"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=23000)]
        )
    except RuntimeError as e:
        print(e)


resnet_model = None


def build_resnet_model(layer_names):
    global resnet_model
    if resnet_model is None:
        resnet = ResNet50(weights="imagenet", include_top=False)
        resnet.trainable = False
        outputs = [resnet.get_layer(name).output for name in layer_names]
        resnet_model = Model(inputs=resnet.input, outputs=outputs)
    return resnet_model


def perceptual_loss(y_true, y_pred):
    resnet_layers = ["conv1_relu", "conv2_block3_out", "conv3_block4_out"]
    resnet_model = build_resnet_model(resnet_layers)

    y_true_lab = tf.concat([tf.zeros_like(y_true[:, :, :, :1]), y_true], axis=-1)
    y_pred_lab = tf.concat([tf.zeros_like(y_pred[:, :, :, :1]), y_pred], axis=-1)

    print("Updated Shape of y_true:", y_true_lab.shape)  # Debug
    print("Updated Shape of y_pred:", y_pred_lab.shape)  # Debug

    y_true_features = resnet_model(y_true_lab)
    y_pred_features = resnet_model(y_pred_lab)

    loss = tf.reduce_sum([tf.reduce_mean(tf.square(f_true - f_pred)) 
                          for f_true, f_pred in zip(y_true_features, y_pred_features)])
    return loss


def build_unet():
    inputs = keras.Input(shape=(224, 224, 1))

    conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    pool1 = layers.MaxPooling2D((2, 2), strides=2)(conv1)

    conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    pool2 = layers.MaxPooling2D((2, 2), strides=2)(conv2)

    conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    pool3 = layers.MaxPooling2D((2, 2), strides=2)(conv3)

    conv4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    pool4 = layers.MaxPooling2D((2, 2), strides=2)(conv4)

    conv5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)


    up4 = layers.Conv2DTranspose(256, (3, 3), strides=2, padding='same', activation='relu')(conv5)
    concat4 = layers.concatenate([up4, conv4])
    conv6 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(concat4)

    up3 = layers.Conv2DTranspose(128, (3, 3), strides=2, padding='same', activation='relu')(conv6)
    concat3 = layers.concatenate([up3, conv3])
    conv7 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(concat3)

    up2 = layers.Conv2DTranspose(64, (3, 3), strides=2, padding='same', activation='relu')(conv7)
    concat2 = layers.concatenate([up2, conv2])
    conv8 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(concat2)

    up1 = layers.Conv2DTranspose(32, (3, 3), strides=2, padding='same', activation='relu')(conv8)
    concat1 = layers.concatenate([up1, conv1])
    conv9 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(concat1)

    conv10 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)
    outputs = layers.Conv2D(2, (1, 1), activation='tanh', padding='same')(conv10)

    model = keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss=perceptual_loss, metrics=['mae'])
    return model


model = build_unet()


model.summary()


def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    img = cv2.resize(img, (224, 224))
    L, A, B = cv2.split(img)

    L = L.astype("float32") / 255.0
    A = (A.astype("float32") - 128) / 128.0
    B = (B.astype("float32") - 128) / 128.0

    return L.reshape(224, 224, 1), np.stack([A, B], axis=-1)


dataset_path = './DATASET_TRAIN'
image_paths = [os.path.join(dataset_path, fname) for fname in os.listdir(dataset_path)]


train_paths, val_paths = train_test_split(image_paths, test_size=0.1, random_state=42)


def data_generator(image_paths, batch_size):
    while True:
        np.random.shuffle(image_paths)
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            X, Y = zip(*[preprocess_image(img) for img in batch_paths])
            yield np.array(X), np.array(Y)


train_gen = data_generator(train_paths, batch_size=16)
val_gen = data_generator(val_paths, batch_size=16)


checkpoint = keras.callbacks.ModelCheckpoint("colorization_model_perceptual_resnet", save_best_only=True, save_format="tf")
reduce_lr = keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6)


start_time = time.time()


history = model.fit(
    train_gen,
    steps_per_epoch=len(train_paths) // 16,
    validation_data=val_gen,
    validation_steps=len(val_paths) // 16,
    epochs=30,
    callbacks=[checkpoint, reduce_lr]
)


def plot_training_history(history):
    epochs = range(1, len(history.history['loss']) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.history['loss'], 'r', label='Training Loss')
    plt.plot(epochs, history.history['val_loss'], 'b', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.history['mae'], 'r', label='Training MAE')
    plt.plot(epochs, history.history['val_mae'], 'b', label='Validation MAE')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.title('Training & Validation MAE')
    plt.legend()

    plt.savefig("colorization_model_perceptual_resnet.svg", format='svg')


plot_training_history(history)


model.save("colorization_model_perceptual_resnet_final", save_format="tf")


end_time = time.time()
training_duration = end_time - start_time
logging.info(f"Total training time: {training_duration:.2f} seconds")