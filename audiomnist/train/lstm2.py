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
    
    # def temporalize(X, y, lookback):
    # output_X = []
    # output_y = []
    # for i in range(len(X)-lookback-1):
    #     t = []
    #     for j in range(1,lookback+1):
    #         # Gather past records upto the lookback period
    #         t.append(X[[(i+j+1)], :])
    #     output_X.append(t)
    #     output_y.append(y[i+lookback+1])
    # return output_X, output_y

    # timesteps = 3
    # X, y = temporalize(X = timeseries, y = np.zeros(len(timeseries)), lookback = timesteps)
    # n_features = 2
    # X = np.array(X)
    # X = X.reshape(X.shape[0], timesteps, n_features)

    '''
    Neural Net model
    '''
    print("building nn")

    timesteps = 8000
    input_dim = 1
    latent_dim = 2000

    def get_model(latent_dim):
        inputs = Input(shape=(timesteps, input_dim))
        encoded = LSTM(latent_dim, return_sequences=False, name="encoder")(inputs)
        decoded = RepeatVector(timesteps)(encoded)
        decoded = LSTM(input_dim, return_sequences=False, name='decoder')(decoded)
        autoencoder = Model(inputs, decoded)
        encoder = Model(inputs, encoded)
        return autoencoder, encoder

    autoencoder, encoder = get_model(latent_dim)

    print(autoencoder.summary())

    adam = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    autoencoder.compile(optimizer=adam, loss='mse', metrics=['accuracy'])

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

    # autoencoder.fit(train_dataset, \
    #             epochs= epochs, \
    #             steps_per_epoch = math.ceil(train_nb_samples/batch_size), \
    #             batch_size = batch_size, \
    #             shuffle=True, \
    #             validation_data=test_dataset, \
    #             validation_steps=math.ceil(test_nb_samples/batch_size), \
    #             callbacks = [tb_callback, checkpoint_callback], \
    #             verbose=1)