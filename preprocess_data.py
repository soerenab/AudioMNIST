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
import multiprocessing
import argparse


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

    pool=multiprocessing.Pool(processes=n_processes)
    _=pool.map(_preprocess_data, [(os.path.join(src, folder), 
                                          os.path.join(dst, folder), 
                                          src_meta) for folder in sorted(folders)])




def _preprocess_data(src_dst_meta):

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
        src_dst_meta: tuple of 3 strings
            Tuple (path to data directory, path to destination directory, path
            to meta file)
    """


    src, dst, src_meta = src_dst_meta

    print("processing {}".format(src))

    metaData = json.load(open(src_meta))

    # create folder for hdf5 files
    if not os.path.exists(dst):
        os.makedirs(dst)
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

        # stft, with seleced parameters, spectrogram will have shape (228,230)
        f, t, Zxx = scipy.signal.stft(embedded_data, 8000, nperseg = 455, noverlap = 420, window='hann')
        # get amplitude
        Zxx = np.abs(Zxx[0:227, 2:-1])
        Zxx = np.atleast_3d(Zxx).transpose(2,0,1)
        # convert to decibel
        Zxx = librosa.amplitude_to_db(Zxx, ref = np.max)
        # save as hdf5 file
        with h5py.File(os.path.join(dst, "AlexNet_{}_{}_{}.hdf5".format(dig, vp, rep)), "w") as f:
            tmp_X = np.zeros([1, 1, 227, 227])

            tmp_X[0, 0] = Zxx
            f['data'] = tmp_X
            f['label'] = np.array([[int(dig), 0 if metaData[vp]["gender"] == "male" else 1]])

        ##### AudioNet #####
        
        embedded_data /= (np.percentile(embedded_data, 95) + 0.001)
        
        with h5py.File(os.path.join(dst, "AudioNet_{}_{}_{}.hdf5".format(dig, vp, rep)), "w") as f:
            tmp_X = np.zeros([1, 1, 1, 8000])

            tmp_X[0, 0, 0] = embedded_data
            f['data'] = tmp_X
            f['label'] = np.array([[int(dig), 0 if metaData[vp]["gender"] == "male" else 1]])

    return



def create_splits(src, dst):

    """
    Creation of text files specifying which files training, validation and test
    set consist of for each cross-validation split. 

    Parameters:
    -----------
        src: string
            Path to directory containing the directories for each subject that
            hold the preprocessed data in hdf5 format.
        dst: string
            Destination where to store the text files specifying training, 
            validation and test splits.

    """

    print("creating splits")
    splits={"digit":{   "train":[   set([28, 56,  7, 19, 35,  1,  6, 16, 23, 34, 46, 53, 36, 57,  9, 24, 37,  2, \
                                          8, 17, 29, 39, 48, 54, 43, 58, 14, 25, 38,  3, 10, 20, 30, 40, 49, 55]),

                                    set([36, 57,  9, 24, 37,  2,  8, 17, 29, 39, 48, 54, 43, 58, 14, 25, 38,  3, \
                                         10, 20, 30, 40, 49, 55, 12, 47, 59, 15, 27, 41,  4, 11, 21, 31, 44, 50]),

                                    set([43, 58, 14, 25, 38,  3, 10, 20, 30, 40, 49, 55, 12, 47, 59, 15, 27, 41, \
                                          4, 11, 21, 31, 44, 50, 26, 52, 60, 18, 32, 42,  5, 13, 22, 33, 45, 51]),

                                    set([12, 47, 59, 15, 27, 41,  4, 11, 21, 31, 44, 50, 26, 52, 60, 18, 32, 42, \
                                          5, 13, 22, 33, 45, 51, 28, 56,  7, 19, 35,  1,  6, 16, 23, 34, 46, 53]),

                                    set([26, 52, 60, 18, 32, 42,  5, 13, 22, 33, 45, 51, 28, 56,  7, 19, 35,  1, \
                                          6, 16, 23, 34, 46, 53, 36, 57,  9, 24, 37,  2,  8, 17, 29, 39, 48, 54])],

                        "validate":[set([12, 47, 59, 15, 27, 41,  4, 11, 21, 31, 44, 50]),
                                    set([26, 52, 60, 18, 32, 42,  5, 13, 22, 33, 45, 51]),
                                    set([28, 56,  7, 19, 35,  1,  6, 16, 23, 34, 46, 53]),
                                    set([36, 57,  9, 24, 37,  2,  8, 17, 29, 39, 48, 54]),
                                    set([43, 58, 14, 25, 38,  3, 10, 20, 30, 40, 49, 55])],

                        "test":[    set([26, 52, 60, 18, 32, 42,  5, 13, 22, 33, 45, 51]),
                                    set([28, 56,  7, 19, 35,  1,  6, 16, 23, 34, 46, 53]),
                                    set([36, 57,  9, 24, 37,  2,  8, 17, 29, 39, 48, 54]),
                                    set([43, 58, 14, 25, 38,  3, 10, 20, 30, 40, 49, 55]),
                                    set([12, 47, 59, 15, 27, 41,  4, 11, 21, 31, 44, 50])]},

            "gender":{  "train":[   set([36, 47, 56, 26, 12, 57, 2, 44, 50, 25, 37, 45]),
                                    set([26, 12, 57, 43, 28, 52, 25, 37, 45, 48, 53, 41]),
                                    set([43, 28, 52, 58, 59, 60, 48, 53, 41, 7, 23, 38]),
                                    set([58, 59, 60, 36, 47, 56, 7, 23, 38, 2, 44, 50])],

                        "validate":[set([43, 28, 52, 48, 53, 41]),
                                    set([58, 59, 60, 7, 23, 38]),
                                    set([36, 47, 56, 2, 44, 50]),
                                    set([26, 12, 57, 25, 37, 45])],

                        "test":[    set([58, 59, 60, 7, 23, 38]),
                                    set([36, 47, 56, 2, 44, 50]),
                                    set([26, 12, 57, 25, 37, 45]),
                                    set([43, 28, 52, 48, 53, 41])]}}

    for split in range(5):
        for modus in ["train", "validate", "test"]:
            for task in ["digit", "gender"]:
                if task == "gender" and split > 3:
                    continue
                with open(os.path.join(dst, "AlexNet_{}_{}_{}.txt".format(task, split, modus)), mode = "w") as txt_file:
                    for vp in splits[task][modus][split]:
                        for filepath in glob.glob(os.path.join(src, "{:02d}".format(vp), "AlexNet*.hdf5")):
                            txt_file.write(filepath+"\n")

                with open(os.path.join(dst, "AudioNet_{}_{}_{}.txt".format(task, split, modus)), mode = "w") as txt_file:
                    for vp in splits[task][modus][split]:
                        for filepath in glob.glob(os.path.join(src, "{:02d}".format(vp), "AudioNet*.hdf5")):
                            txt_file.write(filepath+"\n")




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--source', '-src', default=os.path.join(os.getcwd(), "data"), help="Path to folder containing each participant's data directory.")
    parser.add_argument('--destination', '-dst', default=os.path.join(os.getcwd(), "preprocessed_data"), help="Destination where preprocessed data shall be stored.")
    parser.add_argument('--meta', '-m', default=os.path.join(os.getcwd(), "data", "audioMNIST_meta.txt"), help="Path to meta_information json file.")

    args = parser.parse_args()

    # preprocessing
    preprocess_data(src=args.source, dst=args.destination, src_meta=args.meta)
    # create training, validation and test sets
    create_splits(src=args.destination, dst=args.destination)