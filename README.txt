# audioMNIST

This repository contains a dataset of 30000 audio samples of spoken digits (0-9) of 60 different speakers. 

### Repository structure

-data
Contains one directory per speaker holding the audio recordings. Additionally "audioMNIST_meta.txt" provides meta information such as gender or age of each speaker.

-models
Contains directories of two different model architectures and training parameters in the CAFFE deep learning framework format as well as a bash script to train and test the models.

-recording_scripts
Contains scripts to gather further audio samples. 

-preprocessing_data.py
A python script to preprocess the provided audio records and to store them in a format suitable for the provided caffe models.
