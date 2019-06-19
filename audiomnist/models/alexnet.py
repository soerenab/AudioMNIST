import tensorflow as tf
from tensorflow.keras import layers, initializers, optimizers
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU

#gaussian0_1 = initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None)

def build_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(96, kernel_size = (11,11), input_shape=(227,227,1), strides=(4,4), activation='linear', padding='valid', name='conv1'))
    model.add(LeakyReLU(alpha=0.01,name='leaky_relu1'))
    model.add(layers.MaxPooling2D(pool_size=3,strides=2, name='pool1'))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(256, kernel_size = (5,5), activation='linear', padding='valid', name='conv2'))
    model.add(LeakyReLU(alpha=0.01,name='leaky_relu2'))
    model.add(layers.MaxPooling2D(pool_size=3,strides=2, name='pool2'))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(384, kernel_size = (3,3), activation='relu', padding='valid', name='conv3'))

    model.add(layers.Conv2D(384, kernel_size = (3,3), activation='relu', padding='valid', name='conv4'))

    model.add(layers.Conv2D(256, kernel_size = (3,3), activation='relu', padding='valid', name='conv5'))
    model.add(layers.MaxPooling2D(pool_size=2,strides=2))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(1024, name='dense1'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1024, name='dense2'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, name='dense3', activation='softmax'))

    return model