from ..io.read_dataset import load_audionet_dataset
from audiomnist.train.audionet import split
import tensorflow as tf

from tensorflow.train import RMSPropOptimizer,AdamOptimizer
from tensorflow.keras import losses
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, Lambda
from tensorflow.keras.models import Model
import numpy as np
import pandas as pd
from .splits import splits
import math

import os
import gc
import glob


print("imported")


def train(dataset_path,checkpoint_path, logdir, batch_size, epochs):

    '''
    Load the data
    '''

    tf.enable_eager_execution()
    dataset = load_audionet_dataset(dataset_path)


    def make_tuple(record):
        return (tf.reshape(record['data'],(8000,)),tf.reshape(record['data'],(8000,)))


    
    '''
    Split the dataset
    '''
    train_dataset = dataset.filter(split('digit', 'train')) \
        .map(make_tuple) \
        .shuffle(18000, seed=42) \
        .batch(batch_size) \
        .repeat()

    train_nb_samples = len(splits['digit']['train'][0])*500


    test_dataset = dataset.filter(split('digit', 'test')) \
        .map(make_tuple) \
        .shuffle(10000, seed=42) \
        .batch(batch_size)

    test_nb_samples = len(splits['digit']['test'][0])*500
    
    
    '''
    Neural Net model
    '''
        
    x = Input(shape=(8000,))
    latent_dim = 500
    intermediate_dim = 2000
    original_dim = 8000

    h = Dense(intermediate_dim, activation='relu')(x)
    z_mean = Dense(latent_dim, activation='linear')(h)
    z_log_sigma = Dense(latent_dim, activation='linear', \
                kernel_initializer='zeros', \
                bias_initializer='zeros')(h)

    def sampling(args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=(batch_size, latent_dim))
        return z_mean + K.exp(z_log_sigma/2) * epsilon

    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_sigma])
    decoder_h = Dense(intermediate_dim, activation='relu')
    decoder_mean = Dense(original_dim, activation='sigmoid')
    h_decoded = decoder_h(z)
    x_decoded_mean = decoder_mean(h_decoded)

    # end-to-end autoencoder
    vae = Model(x, x_decoded_mean)
    # encoder, from inputs to latent space
    encoder = Model(x, z_mean)
    # generator, from latent space to reconstructed inputs
    decoder_input = Input(shape=(latent_dim,))
    _h_decoded = decoder_h(decoder_input)
    _x_decoded_mean = decoder_mean(_h_decoded)
    generator = Model(decoder_input, _x_decoded_mean)

    def vae_loss(x, x_decoded_mean):
        xent_loss = K.mean(K.binary_crossentropy(x, x_decoded_mean),axis=1)
        kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=1)
        return xent_loss + kl_loss

    adam = AdamOptimizer(learning_rate=0.001)
    vae.compile(optimizer=adam, loss=vae_loss,metrics = ['accuracy'])

    '''
    Callbacks
    '''
    if not os.path.isdir(logdir): os.mkdir(logdir)
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir,
                                                 batch_size=batch_size)

    if not os.path.isdir(checkpoint_path): os.mkdir(checkpoint_path)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(os.path.join(checkpoint_path, "model.{epoch:02d}-{val_acc:.2f}"),
                                                            save_weights_only=True)

    gc_callback = tf.keras.callbacks.LambdaCallback(on_batch_end=lambda batch,_: gc.collect())

    
    '''
    Fit the model
    '''
    vae.fit(train_dataset, \
                epochs=epochs, \
                steps_per_epoch = math.ceil(train_nb_samples/batch_size), \
                batch_size = batch_size, \
                shuffle=True, \
                validation_data=test_dataset, \
                validation_steps=math.ceil(test_nb_samples/batch_size), \
                callbacks = [tb_callback, checkpoint_callback])
    
