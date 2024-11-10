import tensorflow as tf

from settings.general_variables import *


def create_model_210720(kernel_sizes, input_train_shape):
    """
    指定されたカーネルサイズのリストに基づいてCNNモデルを生成する。

    Args:
        kernel_sizes: カーネルサイズのリスト（list of int）
        input_train_shape: 学習用入力データのshape
    Returns:
        生成されたモデル（tensorflow.keras.models.Model）
    """
    inputs = tf.keras.layers.Input(shape=input_train_shape[1:])

    x = inputs
    for kernel_size in kernel_sizes:
        x = tf.keras.layers.Conv1D(16, kernel_size, kernel_initializer=tf.keras.initializers.HeNormal(seed=40))(x)  # 畳み込み層
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.5, seed=40)(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(len(WIND_ENVIRONMENTS), kernel_initializer=tf.keras.initializers.HeNormal(seed=40))(x)
    outputs = tf.keras.activations.softmax(x)

    return tf.keras.models.Model(inputs, outputs)


def create_simple_dense_model_231110(input_train_shape):
    """
    Creates a simple fully connected neural network model for 3D input data.

    Args:
        input_train_shape: Shape of the input data (tuple of 3 integers).

    Returns:
        A TensorFlow Keras model with fully connected layers.
    """
    inputs = tf.keras.layers.Input(input_train_shape[1:])

    # Flatten the input data to make it suitable for Dense layers
    x = tf.keras.layers.Flatten()(inputs)

    # Fully connected layers
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    # Output layer
    outputs = tf.keras.layers.Dense(len(WIND_ENVIRONMENTS), activation='softmax')(x)

    return tf.keras.models.Model(inputs=inputs, outputs=outputs)


