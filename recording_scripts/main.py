# -*- coding: utf-8 -*-

import sounddevice as sd
import numpy as np
import datetime
import os
import argparse

import util


def run(noMic, noTest, auto_cut):

    """
    Main routine that will start the recording session.
    Parameters:
    -----------
        noMic: boolean
            Select true if no microphone is connected to the computer. If True,
            no recordings will be saved.
        noTest: boolean
            Select True to disable the initial test run of the recording 
            session.
    """


    #Initialize basic recording settings
    # original samplerate: 48000
    SETTINGS = {'samplerate'    :  48000,\
                'channels'      :  1,\
                'seconds'       :  1,\
                'pause'         :  1,\
                'device_name'   :  "RODE*", \
                'device_index'  : -1,\
                'recording_log' : './session_log.txt',\
                'data_folder'   : './data',\
                'meta_file'     : './meta.txt'
    }
    SETTINGS['device_index'] = util.find_input_device(SETTINGS['device_name'])
    #IMPORTANT! !!
    #If you are recording with the RODE USB-NT Microphone and are having issues with the audio output jack
    #on the mic itself, make sure that in the ubuntu sound settings, the analog output meter is maxed out.
    #for one, the digital output volume control is tied to the audio volume control and second, the output is muted
    #completely when setting the analog audio to below ~20% volume
    #
    #if this does not help, try disabling the usb auto power suspension for your connected devices.
    #see http://askubuntu.com/a/301416

    #Initialize the to be recorded samples in random order. DO NOT CHANGE THIS.
    DIGITS = 10
    REPS = 2
    SAMPLES     = np.arange(DIGITS).repeat(REPS) # each digit , many times
    REPETITION  = np.array([range(REPS)]).repeat(DIGITS, axis=0).flatten() #for each digit the index of its repetition

    #Shuffle samples
    I = np.random.permutation(SAMPLES.size)
    SAMPLES = [(SAMPLES[i],REPETITION[i]) for i in I] # [(digit, rep)* ]

    #Initialize subject dummy
    SUBJECT = { 'alias'         : "johndoe",\
                'age'           : 1234,\
                'origin'        : "Europe, Germany, Berlin",\
                'accent'        : "German",\
                'native speaker': ['no', 'yes'], \
                'gender'        : ['male', 'female', 'other'],\
                'recordingroom' : "<enter place where recording was made>",\
                'recordingdate' : datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S')
                }   # if these dictionary keys are modified, 
                    # make sure that utils.is_subject_in_meta() 
                    #is adapted accordingly

    # When the script is executed the first time the data folder might be missing.
    if not os.path.exists(SETTINGS["data_folder"]):
       os.mkdir(SETTINGS["data_folder"])
    
    if not os.path.exists(SETTINGS["meta_file"]):
        with open(SETTINGS["meta_file"], "w+") as mf:
            mf.write("{}\n")

    #start session
    util.start_recording_session(SUBJECT, SETTINGS, SAMPLES, noMic, noTest, auto_cut)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--noMic', dest='noMic', action='store_true', 
                        help="Select this option to have a dry-run of the "\
                             "experiment, for instance when no microphone is "\
                             "connected to the computer. No recordings will be "\
                             "made.")
    
    parser.add_argument('--noTest', dest='noTest', action='store_true', 
                        help="Select this option to disable the initial test "\
                              "run before the actual recordings start.")

    parser.add_argument('--auto_cut', action='store_true', 
                        help="Try to cut recorded digit series into recordings "\
                             "of individual (i.e. single) digits. Repeat any "\
                             "series that could not be cut automatically. "\
                             "(This feature is not very reliable and it is "\
                             "recommended to manually cut digit series instead. "\
                             "Therefore it is turned off by default.)")
    
    parser.set_defaults(noMic=False)
    parser.set_defaults(noTest=False)
    args = parser.parse_args()

    run(args.noMic, args.noTest, args.auto_cut)