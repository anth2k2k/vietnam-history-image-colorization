import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time
import logging


os.environ['CUDA_VISIBLE_DEVICES'] = "0"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=23000)]  # Giới hạn 20GB
        )
    except RuntimeError as e:
        print(e)


def mse_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))


def build_unet():
    inputs = keras.Input(shape=(224, 224, 1))

    # Encoder
    conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    pool1 = layers.MaxPooling2D((2, 2), strides=2)(conv1)

    conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    pool2 = layers.MaxPooling2D((2, 2), strides=2)(conv2)

    conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    pool3 = layers.MaxPooling2D((2, 2), strides=2)(conv3)

    conv4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    pool4 = layers.MaxPooling2D((2, 2), strides=2)(conv4)

    conv5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)

    # Decoder
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
    model.compile(optimizer='adam', loss=mse_loss, metrics=['mae'])
    return model


model = build_unet()


model.summary()


def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    img = cv2.resize(img, (224, 224))
    L, A, B = cv2.split(img)

    L = L.astype("float32") / 255.0  # Normalize L
    A = (A.astype("float32") - 128) / 128.0  # Normalize A, B to [-1,1]
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


checkpoint = keras.callbacks.ModelCheckpoint("colorization_model_mse", save_best_only=True, save_format="tf")
reduce_lr = keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6)

# Đo thời gian bắt đầu
start_time = time.time()

# Train model and save history
history = model.fit(
    train_gen,
    steps_per_epoch=len(train_paths) // 16,
    validation_data=val_gen,
    validation_steps=len(val_paths) // 16,
    epochs=30,
    callbacks=[checkpoint, reduce_lr]
)


# Plot training history
def plot_training_history(history):
    epochs = range(1, len(history.history['loss']) + 1)

    plt.figure(figsize=(12, 5))

    # Biểu đồ Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.history['loss'], 'r', label='Training Loss')
    plt.plot(epochs, history.history['val_loss'], 'b', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('[MSE] Training & Validation Loss')
    plt.legend()

    # Biểu đồ MAE
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.history['mae'], 'r', label='Training MAE')
    plt.plot(epochs, history.history['val_mae'], 'b', label='Validation MAE')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.title('[MSE] Training & Validation MAE')
    plt.legend()

    plt.savefig("colorization_model_mse.svg", format='svg')


# Hiển thị kết quả training
plot_training_history(history)


# Save final model using TensorFlow format
model.save("colorization_model_mse_final", save_format="tf")

end_time = time.time()
training_duration = end_time - start_time
# Ghi thời gian chạy vào file log
logging.info(f"Total training time: {training_duration:.2f} seconds")

