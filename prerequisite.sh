#!/bin/bash

echo "Setup TensorFlow environment..."

python3 -m venv tf
echo 'export TF_USE_LEGACY_KERAS=1' >> tf/bin/activate
source tf/bin/activate
pip install --upgrade pip
pip install tensorflow[and-cuda] opencv-python tf_keras gdown matplotlib scikit-learn
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
python -c "from tensorflow import keras; print(keras.__path__)"

echo "Done!"


echo "Setup dataset..."

gdown https://drive.google.com/uc?id=1OnicSUsL7UlKL0TRxj3ydXMQn9WATS-t
gdown https://drive.google.com/uc?id=1I316OnbVqXZQyIKq_-i9WoK-yZU0rKYR
unzip DATASET_TRAIN.zip -d .
unzip DATASET_TEST.zip -d .

echo "Done!"
