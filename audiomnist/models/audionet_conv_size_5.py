import tensorflow as tf
from tensorflow.keras import layers, initializers, optimizers

#gaussian0_1 = initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None)

def build_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv1D(100, 5, input_shape= (8000, 1), padding='same', activation='relu', name='conv1'))
    model.add(layers.MaxPooling1D(pool_size=2, strides=2, name='pool1'))

    model.add(layers.Conv1D(64, 5, padding='same', activation='relu', name='conv2'))
    model.add(layers.MaxPooling1D(pool_size=2, strides=2, name='pool2'))

    model.add(layers.Conv1D(128, 5, padding='same', activation='relu', name='conv3'))
    model.add(layers.MaxPooling1D(pool_size=2, strides=2, name='pool3'))

    model.add(layers.Conv1D(128, 5, padding='same', activation='relu', name='conv4'))
    model.add(layers.MaxPooling1D(pool_size=2, strides=2, name='pool4'))

    model.add(layers.Conv1D(128, 5, padding='same', activation='relu', name='conv5'))
    model.add(layers.MaxPooling1D(pool_size=2, strides=2, name='pool5'))

    model.add(layers.Conv1D(128, 5, padding='same', activation='relu', name='conv6'))
    model.add(layers.MaxPooling1D(pool_size=2, strides=2, name='pool6'))

    model.add(layers.Flatten())
    model.add(layers.Dense(1024, name='dense1'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, name='dense2'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, name='dense3', activation='softmax'))

    return model