import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from tensorflow import keras
from keras.applications import VGG16, ResNet50
from keras.models import Model

vgg_model = None
resnet_model = None

def mse_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

def build_vgg_model(layer_names):
    global vgg_model
    if vgg_model is None:
        vgg = VGG16(weights="imagenet", include_top=False)
        vgg.trainable = False
        outputs = [vgg.get_layer(name).output for name in layer_names]
        vgg_model = Model(inputs=vgg.input, outputs=outputs)
    return vgg_model

def build_resnet_model(layer_names):
    global resnet_model
    if resnet_model is None:
        resnet = ResNet50(weights="imagenet", include_top=False)
        resnet.trainable = False
        outputs = [resnet.get_layer(name).output for name in layer_names]
        resnet_model = Model(inputs=resnet.input, outputs=outputs)
    return resnet_model

def perceptual_loss_vgg(y_true, y_pred):
    vgg_layers = ["block1_conv2", "block2_conv2", "block3_conv3"]
    vgg_model = build_vgg_model(vgg_layers)

    y_true_lab = tf.concat([tf.zeros_like(y_true[:, :, :, :1]), y_true], axis=-1)
    y_pred_lab = tf.concat([tf.zeros_like(y_pred[:, :, :, :1]), y_pred], axis=-1)

    y_true_features = vgg_model(y_true_lab)
    y_pred_features = vgg_model(y_pred_lab)

    loss = tf.reduce_sum([tf.reduce_mean(tf.square(f_true - f_pred))
                          for f_true, f_pred in zip(y_true_features, y_pred_features)])
    return loss

def perceptual_loss_resnet(y_true, y_pred):
    resnet_layers = ["conv1_relu", "conv2_block3_out", "conv3_block4_out"]
    resnet_model = build_resnet_model(resnet_layers)

    y_true_lab = tf.concat([tf.zeros_like(y_true[:, :, :, :1]), y_true], axis=-1)
    y_pred_lab = tf.concat([tf.zeros_like(y_pred[:, :, :, :1]), y_pred], axis=-1)

    y_true_features = resnet_model(y_true_lab)
    y_pred_features = resnet_model(y_pred_lab)

    loss = tf.reduce_sum([tf.reduce_mean(tf.square(f_true - f_pred))
                          for f_true, f_pred in zip(y_true_features, y_pred_features)])
    return loss

def preprocess_image(image_path, target_size=(224, 224)):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    original_size = img.shape[:2]
    L, A, B = cv2.split(img)
    L_resized = cv2.resize(L, target_size)

    L_resized = L_resized.astype("float32") / 255.0
    A_resized = (cv2.resize(A, target_size).astype("float32") - 128) / 128.0
    B_resized = (cv2.resize(B, target_size).astype("float32") - 128) / 128.0

    return L_resized.reshape(224, 224, 1), np.stack([A_resized, B_resized], axis=-1), L, original_size

def colorize_image(model, image_path):
    L_resized, _, L_original, original_size = preprocess_image(image_path)
    L_input = np.expand_dims(L_resized, axis=0)

    ab_pred = model.predict(L_input)[0]
    ab_pred = (ab_pred * 128 + 128).astype("uint8")

    ab_pred_resized = cv2.resize(ab_pred, (original_size[1], original_size[0]), interpolation=cv2.INTER_CUBIC)

    colorized_img = cv2.merge([L_original, ab_pred_resized[:, :, 0], ab_pred_resized[:, :, 1]])
    colorized_img = cv2.cvtColor(colorized_img, cv2.COLOR_LAB2BGR)
    colorized_img = cv2.cvtColor(colorized_img, cv2.COLOR_BGR2RGB)

    return colorized_img

@st.cache_resource
def load_models():
    mse_model = tf.keras.models.load_model("models/colorization_model_mse", custom_objects={"mse_loss": mse_loss})
    vgg_model = tf.keras.models.load_model("models/colorization_model_perceptual",
                                           custom_objects={"perceptual_loss": perceptual_loss_vgg})
    resnet_model = tf.keras.models.load_model("models/colorization_model_perceptual_resnet",
                                              custom_objects={"perceptual_loss": perceptual_loss_resnet})
    return mse_model, vgg_model, resnet_model

# Streamlit
st.title("Vietnam History Image Colorization - by @tanh2k2k")

uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    original_filename = uploaded_file.name

    with open(original_filename, "wb") as f:
        f.write(uploaded_file.read())

    st.image(original_filename, caption="Upload sucessfully", use_container_width=True)

    # Load models
    with st.spinner("Colorizing..."):
        mse_model, vgg_model, resnet_model = load_models()

        # Chạy colorization với 3 model
        result_mse = colorize_image(mse_model, original_filename)
        result_vgg = colorize_image(vgg_model, original_filename)
        result_resnet = colorize_image(resnet_model, original_filename)

    # Hiển thị ảnh kết quả
    st.subheader("Colorized Results")

    col1, col2, col3 = st.columns(3)
    col1.image(result_mse, caption="MSE Model", use_container_width=True)
    col2.image(result_vgg, caption="Perceptual VGG Model", use_container_width=True)
    col3.image(result_resnet, caption="Perceptual ResNet Model", use_column_width=True)

    # Nút tải về
    st.download_button(f"Download - MSE", data=cv2.imencode('.jpg', cv2.cvtColor(result_mse, cv2.COLOR_RGB2BGR))[1].tobytes(), file_name=f"{original_filename.split('.')[0]}_mse.jpg", mime="image/jpeg")
    st.download_button(f"Download - VGG", data=cv2.imencode('.jpg', cv2.cvtColor(result_vgg, cv2.COLOR_RGB2BGR))[1].tobytes(), file_name=f"{original_filename.split('.')[0]}_vgg.jpg", mime="image/jpeg")
    st.download_button(f"Download - ResNet", data=cv2.imencode('.jpg', cv2.cvtColor(result_resnet, cv2.COLOR_RGB2BGR))[1].tobytes(), file_name=f"{original_filename.split('.')[0]}_resnet.jpg", mime="image/jpeg")