from ..io.read_dataset import load_audionet_dataset
from audiomnist.train.audionet import split
import tensorflow as tf

from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, LSTM, RepeatVector
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
    print("loading data")
    tf.enable_eager_execution()
    dataset = load_audionet_dataset(dataset_path)


    '''
    Compute max for standardization
    '''

    print("compute max")
    maxi = 0
    for e in dataset :
        l = e['data'].numpy().flatten()
        maxi = max(max(max(l),abs(min(l))),maxi)
        
    def make_tuple(record):
        return (tf.reshape(record['data'],(8000,1)),tf.reshape(record['data'],(8000,1)))


    
    '''
    Split the dataset
    '''

    print("splitting data")
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
    print("building nn")

    timesteps = 1
    input_dim = 8000
    latent_dim = 100

    inputs = Input(shape=(timesteps, input_dim))
    encoded = LSTM(latent_dim)(inputs)

    decoded = RepeatVector(timesteps)(encoded)
    decoded = LSTM(input_dim, return_sequences=True)(decoded)

    sequence_autoencoder = Model(inputs, decoded)
    encoder = Model(inputs, encoded)

    # sequence_autoencoder.compile(loss='mean_squared_error', optimizer='adam')

    # print(train_dataset)
    # inputs = Input(shape=(None,8000))
    # encoded = LSTM(None, 1000)(inputs)

    # decoded = LSTM((None, 8000), return_sequences=True)(decoded)

    # sequence_autoencoder = Model(inputs, decoded)
    # encoder = Model(inputs, encoded)
    
    # adam = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    # sequence_autoencoder.compile(optimizer=adam, loss='mse', metrics=['accuracy'])

    sequence_autoencoder.compile(loss='mean_squared_error', optimizer='adam')

    print(sequence_autoencoder.summary())
    

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
    print("fitting")
    encoder.fit(train_dataset, \
                epochs=epochs, \
                steps_per_epoch = math.ceil(train_nb_samples/batch_size), \
                batch_size = batch_size, \
                shuffle=True, \
                validation_data=test_dataset, \
                validation_steps=math.ceil(test_nb_samples/batch_size), \
                callbacks = [tb_callback, checkpoint_callback]
    )
    
