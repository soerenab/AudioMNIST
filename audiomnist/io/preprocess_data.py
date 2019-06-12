# -*- coding: utf-8 -*-

import numpy as np
import glob
import os
import sys
import scipy.io.wavfile as wavf
import scipy.signal
import h5py
import json
import librosa
import tensorflow as tf
from tqdm import tqdm

tf.enable_eager_execution()

# The following functions can be used to convert a value to a type compatible
# with tf.Example.

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def preprocess_data(src, dst, src_meta, n_processes=15):

    """
    Calls for distibuted preprocessing of the data.

    Parameters:
    -----------
        src: string
            Path to data directory.
        dst: string
            Path to directory where preprocessed data shall be stored.
        stc_meta: string
            Path to meta_information file.
        n_processes: int
            number of simultaneous processes to use for data preprocessing.
    """

    folders = []

    for folder in os.listdir(src):
        # only process folders
        if not os.path.isdir(os.path.join(src, folder)):
            continue
        folders.append(folder)

    audionet_writer=tf.python_io.TFRecordWriter(os.path.join(dst, "audionet.tfrecords"))
    alexnet_writer=tf.python_io.TFRecordWriter(os.path.join(dst, "alexnet.tfrecords"))
    for folder in tqdm(sorted(folders)):
        _preprocess_data(os.path.join(src, folder), 
                        audionet_writer,
                        alexnet_writer, 
                        src_meta)




def _preprocess_data(src, audionet_writer, alexnet_writer, src_meta):

    """
    Preprocessing for all data files in given directory.
    Preprocessing includes:

        AlexNet: resampling to 8000 Hz, 
                 embedding in zero vector, 
                 transformation to amplitute spectrogram representation in dB.
        
        AudioNet: resampling to 8000 Hz, 
                  embedding in zero vector, 
                  normalization at 95th percentile.

    Preprocessed data will be stored in hdf5 files with one datum per file.
    In terms of I/O, this is not very efficient but it allows to easily change
    training, validation, and test sets without re-preprocessing or redundant 
    storage of preprocessed files.

    Parameters:
    -----------
        src_writer_meta: tuple of 3 strings
            Tuple (path to data directory, path to destination directory, path
            to meta file)
    """

    metaData = json.load(open(src_meta))

    # loop over recordings
    for filepath in sorted(glob.glob(os.path.join(src, "*.wav"))):

        # infer sample info from name
        dig, vp, rep = filepath.rstrip(".wav").split("/")[-1].split("_")
        # read data
        fs, data = wavf.read(filepath)
        # resample
        data = librosa.core.resample(y=data.astype(np.float32), orig_sr=fs, target_sr=8000, res_type="scipy")
        # zero padding
        if len(data) > 8000:
            raise ValueError("data length cannot exceed padding length.")
        elif len(data) < 8000:
            embedded_data = np.zeros(8000)
            offset = np.random.randint(low = 0, high = 8000 - len(data))
            embedded_data[offset:offset+len(data)] = data
        elif len(data) == 8000:
            # nothing to do here
            embedded_data = data
            pass

        ##### AlexNet #####

        # stft, with seleced parameters, spectrogram will have shape (227,227)
        f, t, Zxx = scipy.signal.stft(embedded_data, 8000, nperseg = 455, noverlap = 420, window='hann')
        # get amplitude
        Zxx = np.abs(Zxx[0:227, 2:-1])
        Zxx = np.atleast_3d(Zxx).transpose(2,0,1)
        # convert to decibel
        Zxx = librosa.amplitude_to_db(Zxx, ref = np.max)
        # save as hdf5 file
        tmp_X = np.zeros([1, 227, 227, 1])
        tmp_X[0, :, :, 0] = Zxx

        alexnet_feature = {
            'data' : tf.train.Feature(float_list=tf.train.FloatList(value=tmp_X.ravel())),
            'shape' : tf.train.Feature(float_list=tf.train.FloatList(value=tmp_X.shape)),
            'digit' : _int64_feature(int(dig)),
            'gender' : _int64_feature(0 if metaData[vp]["gender"] == "male" else 1),
            'vp' : _int64_feature(int(vp))
        }

        alexnet_example_proto = tf.train.Example(features=tf.train.Features(feature=alexnet_feature))

        alexnet_writer.write(alexnet_example_proto.SerializeToString())

        ##### AudioNet #####
        
        embedded_data /= (np.percentile(embedded_data, 95) + 0.001)
        
        tmp_X = np.zeros([1, 8000, 1, 1])

        tmp_X[0, :, 0, 0] = embedded_data
        audionet_feature = {
            'data' : tf.train.Feature(float_list=tf.train.FloatList(value=tmp_X.ravel())),
            'shape' : tf.train.Feature(float_list=tf.train.FloatList(value=tmp_X.shape)),
            'digit' : _int64_feature(int(dig)),
            'gender' : _int64_feature(0 if metaData[vp]["gender"] == "male" else 1),
            'vp' : _int64_feature(int(vp))
        }

        audionet_example_proto = tf.train.Example(features=tf.train.Features(feature=audionet_feature))

        audionet_writer.write(audionet_example_proto.SerializeToString())

    return

