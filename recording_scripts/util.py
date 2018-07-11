# -*- coding: utf-8 -*-
import sounddevice as sd
import fnmatch
import json
import os
import sys
import datetime
import fileinput
from psychopy import visual
import glob
import scipy.signal
from scipy.io import wavfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import recording_gui as rgui

# ------------------
# AUDIO STUFF
# ------------------

def start_recording_session(SUBJECT, SETTINGS, SAMPLES, noMic, noTest, auto_cut):

    """
    Creates and starts audio recording setup.
    Parameters:
    -----------
        SUBJECT: dict
        SETTINGS: dict
        SAMPLES: dict
        noMic: bool
            Select True if no microphone is connected. No records will be made.
        noTest: bool
            If True, there will be no initial test run.

        See main.py for examples of SUBJECT, SETTINGS, SAMPLES.
    """

    if noMic:
        if not rgui.SubjectMessageScreen('Found "-noMic" flag: No recordings will be made.').show():
            print('User cancelled.')
            quit()
    metadata = read_meta(SETTINGS)
    activeSubject = read_checkpoint()

    if activeSubject and rgui.SubjectMessageScreen('Found an active session. Continue?').show():    
        SUBJECT = activeSubject
        print('continuing session.')
        SAMPLES = get_missing_samples_for_subject(SETTINGS, SUBJECT, SAMPLES)
    else:

        print('new session.')
        SUBJECT['alias'] = 'tmp_'+str(datetime.datetime.now().strftime("%d-%m-%y-%H-%M-%s"))
        while is_subject_in_metadata(SUBJECT, metadata) != False: # in case tmp_str(time.time()) is for some reason already in the data
            SUBJECT['alias'] = 'tmp_'+str(datetime.datetime.now().strftime("%d-%m-%y-%H-%M-%s"))
        
        if activeSubject: # in case there is a checkpoint but that session shell not be continued.
            disable_checkpoint()
        # make new checkpoint for this "fresh" session
        write_checkpoint(SUBJECT)

    win = visual.Window(color = (-0.5, -0.5, -0.5), 
                        fullscr = True, 
                        allowGUI = True, 
                        units = 'norm',
                        name = 'Spoken numbers recording',
                        waitBlanking = False)

    dataRecScreen = rgui.DataRecordingScreen(SUBJECT, SETTINGS, SAMPLES, window=win, auto_cut=auto_cut)

    if not noTest:
        dataRecScreen.show(istest = True, noMic = noMic)

    # store the name that is used to write out .wav files during recording. It is needed to update the filenames once a subject has entered its personal data.
    subjectFileIdentifier = subject_filename_convention(SUBJECT['alias'])
    #do the actual recording in this UI
    samplesToRepeat = dataRecScreen.show(istest = False, noMic = noMic)
    dataRecScreen.close()
    while len(samplesToRepeat) > 0:
        rgui.SubjectMessageScreen('{} digits could not be cut and need to be repeated'.format(len(samplesToRepeat))).show()
        win = visual.Window(color = (-0.5, -0.5, -0.5), 
                        fullscr = True, 
                        allowGUI = True, 
                        units = 'norm',
                        name = 'Spoken numbers recording',
                        waitBlanking = False)
        dataRecScreen = rgui.DataRecordingScreen(SUBJECT, SETTINGS, samplesToRepeat, window=win)
        samplesToRepeat = dataRecScreen.show(istest = False, noMic = noMic)
        dataRecScreen.close()
    

    # ask subject to enter personal data
    session_initialized = False
    SUBJECT_dialog = SUBJECT.copy()
    while not session_initialized:
        SUBJECT = SUBJECT_dialog.copy()
        subjectDataScreen = rgui.SubjectDataScreen(SUBJECT)
        while not subjectDataScreen.OK:
            print('User pressed "cancel" (but this has no effect.)')
            SUBJECT = SUBJECT_dialog.copy()
            subjectDataScreen = rgui.SubjectDataScreen(SUBJECT)
        
        SUBJECT = {k: unicode(v).encode("utf-8") for k,v in SUBJECT.iteritems()}

        SUBJECT['alias'] = subject_filename_convention(SUBJECT['alias'])
        # ask for subject status
        subject_in_meta = is_subject_in_metadata(SUBJECT,metadata)

        if subject_in_meta == False: # totally new subject. continue as is. you are fine.
            # add subject data to meta file
            add_to_meta(metadata, SUBJECT)
            write_meta(metadata, SETTINGS)
            update_subject_filenames(subjectFileIdentifier, SUBJECT['alias'], SETTINGS)
            session_initialized = True
        else:
            print('This alias is already taken!')
            rgui.SubjectMessageScreen('This alias is already taken. Please enter a new alias.').show()
            
    #termination screen and shutdown
    remove_checkpoint()
    rgui.SubjectMessageScreen('Done! Thank you!').show()

    return



def find_input_device(namingpattern='*'):
    #
    """
    Finds a suitable input device folliwing an input naming pattern
    """
    device_index = -1
    devices = sd.query_devices()

    for i in xrange(len(devices)):
        print devices[i]['name'], devices[i]['max_input_channels']

        if devices[i]['max_input_channels'] > 0 and fnmatch.fnmatch(devices[i]['name'],namingpattern):
            print '\nfound at least one matching device!'
            print 'selecting at index',i,':', devices[i]['name'],'(this is the first pick)'
            device_index = i
            break

    if device_index < 0:
        print 'WARNING! No suitable sound device found!'
    return device_index



# -----------------
# META DATA IO
# -----------------

def read_meta(SETTINGS):
    #reads the metadata json dictionary from disk.
    print 'reading metadata from', SETTINGS['meta_file']
    with open(SETTINGS['meta_file'],'rb') as f:
        metadata = json.loads(f.read())
        return metadata



def add_to_meta(metadata, SUBJECT):
    #raises exception
    #adds the data of SUBJECT to the metadata file, returns the extended data
    if SUBJECT['alias'] in metadata: #safety catch
        raise Exception('Another subject of name {} does already exist in meta data file {}. Pick a new, unique identifier!'.format(SUBJECT['name'], SETTINGS['meta_files']))
    else:
        #metadata[SUBJECT['alias']] = {'age':SUBJECT['age'], 'gender':SUBJECT['gender'],'accent':SUBJECT['accent'],'recordingdate':SUBJECT['recordingdate']}
        tmp_SUBJECT = SUBJECT.copy()
        tmp_SUBJECT.pop('alias')
        metadata[SUBJECT['alias']] = tmp_SUBJECT
        return metadata


def write_meta(metadata,SETTINGS):
    #writes the given metadata dictionary to the file name specified in SETTINGS

    with open(SETTINGS['meta_file'],'wb') as f:
        f.write(json.dumps(metadata, sort_keys=True, indent=4))


# ------------------------
# DATA COMPLETENESS CHECKS
# ------------------------
def is_subject_in_metadata(SUBJECT, meta):
    #returns True if subject exists as given
    #returns 'alias' if name is in data, but other data differs.
    #returns False if subject does not even exist by name
    subname = SUBJECT['alias']
    if subname in meta:
        if meta[subname]['age'] == SUBJECT['age']\
            and meta[subname]['gender'] == SUBJECT['gender']\
            and meta[subname]['origin'] == SUBJECT['origin']\
            and meta[subname]['native speaker'] == SUBJECT['native speaker']:
                return True
        else:
            return 'alias'
    else:
        return False


def get_missing_samples_for_subject(SETTINGS,SUBJECT,SAMPLES):
    #checks the data for the given subject and returns those files which are missing.
    #samples is a list of tuples (digit,repetition)
    missing = []
    if 'digitSeries' in SETTINGS.keys() and SETTINGS['digitSeries']:
        thisSubjectsRecs = glob.glob(os.path.join(SETTINGS['data_folder'], SUBJECT["alias"])+"*")
        if len(thisSubjectsRecs) == 0:
            print("Found no recordings for subject")
            missing = SAMPLES
        else:
            print("Found recordings: {}".format(thisSubjectsRecs))
            for d,r in SAMPLES:
                tmp = "__"+str(d)+"_"+str(r)
                thisSubjectsRecs_string = "".join(thisSubjectsRecs)
                if not tmp + "__" in thisSubjectsRecs_string and \
                   not tmp + "." in thisSubjectsRecs_string:
                    missing.append((d,r))
                    

    else:
        for d,r in SAMPLES:
            wavpath = get_full_sample_path(SETTINGS, SUBJECT, d, r)
            if not os.path.isfile(wavpath):
                missing.append((d,r))



    return missing


def get_full_sample_path(SETTINGS, SUBJECT, digit, repetition):
    # for a given SUBJECT, digit and its repetion, return the full output file path according to the data path in SETTINGS
    filename = str(digit) + '_' + subject_filename_convention(SUBJECT['alias']) + '_' + str(repetition) + '.wav'
    return os.path.join(SETTINGS['data_folder'], filename)


def get_full_series_path(SETTINGS, SUBJECT, sampleSeries):
    filename = subject_filename_convention(SUBJECT['alias']) + "".join(["__"+str(d)+"_"+ str(r) for d, r in sampleSeries]) + '.wav'
    return os.path.join(SETTINGS['data_folder'], filename)


# -----------------------
# .wav file name handling
# -----------------------

def subject_filename_convention(subjectname):
    return subjectname.lower()


def update_subject_filenames(originalName, newName, SETTINGS):
    
    """
    To avoid any potential biases, subject enter personal information after the
    experiment. This requires updating the filename once they have input their 
    alias (e.g. '01').
    Parameters:
    -----------
        originalName: string
        newName: string
    """
    
    path = SETTINGS['data_folder']
    dataCounter = 0
    # update filenames
    for filename in os.listdir(path):
        if filename.endswith('.wav') and originalName in filename:
            origName = os.path.join(path, filename)
            os.rename(origName, origName.replace(originalName, newName))
            dataCounter += 1
    print('Updated {:d} filenames.'.format(dataCounter))

    pdfCounter = 0
    for filename in os.listdir(path):
        if filename.endswith('.pdf') and originalName in filename:
            origName = os.path.join(path, filename)
            os.rename(origName, origName.replace(originalName, newName))
            pdfCounter += 1
    print('Updated {:d} pdf filenames.'.format(pdfCounter))

    cutPath = os.path.join(SETTINGS['data_folder'], "cut")
    cutCounter = 0
    if os.path.exists(cutPath):
        for filename in os.listdir(cutPath):
            if filename.endswith('.wav') and originalName in filename:
                origName = os.path.join(cutPath, filename)
                os.rename(origName, origName.replace(originalName, newName))
                cutCounter += 1
        print('Updated {:d} cut filenames.'.format(cutCounter))

    # update names in logfile
    path = SETTINGS['recording_log']
    logCounter = 0
    for line in fileinput.input(path, inplace = True):
        if originalName in line:
            newLine = line.replace(originalName, newName)
            print "%s" % (newLine),
            logCounter += 1
        else:
            print "%s" % (line),

    print('Updated {:d} lines in logfile.'.format(logCounter))
    if not dataCounter == logCounter:
        print("Something appears to be wrong: the number of updated names in the log file should equal the number of updated .wav files in the data folder.")
    

def remove_recordings(filesToRemove, SETTINGS):

    """
    Method to remove files.
    Parameters:
        filesToRemove: list of strings
            filenames to be removed
        SETTINGS: dict, see main.py
    """

    path = SETTINGS['data_folder']
    dataCounter = 0
    # Remove files
    for filename in filesToRemove:
        print('Removing '+filename)
        os.remove(filename)
        dataCounter += 1
    print('Removed {:d}/{:d}.'.format(dataCounter, len(filesToRemove)))

# --------------------------------------------------
# checkpoint functions for quick restart after crash
# --------------------------------------------------

def read_checkpoint(path = './activeRecording.checkpoint'):
    # returns content of '../activeRecording.checkpoint' if that file exists and None otherwise
    if os.path.exists(path):
        print 'found an active session'
        with open(path,'rb') as f:
            return json.loads(f.read())
    else:
        return None


def write_checkpoint(subject, path = './activeRecording.checkpoint'):
    # creates a checkpoint file that contains the data in subject (where subject should be a dictionary)
    with open(path,'wb') as f:
        f.write(json.dumps(subject, sort_keys=True, indent=4))


def remove_checkpoint(path = './activeRecording.checkpoint'):
    os.remove(path)


def disable_checkpoint(path = './activeRecording.checkpoint'):
    if os.path.exists(path):
        with open(path,'rb') as f:
            sess_id = json.loads(f.read())['alias']
        os.rename(path, path.replace('active', 'inactive')+'_'+sess_id)
        print('existing checkpoint disabled')


# ------------------------
# RECORDING LOG FUNCTIONS
# ------------------------

def add_to_log(SETTINGS, wavpath):
    logpath = SETTINGS['recording_log']
    if not os.path.isfile(logpath):
        with open(logpath, 'wb') as f:
            f.write(wavpath + '\n')
    else:
        with open(logpath,'ab') as f:
            f.write(wavpath + '\n')

def log_to_list(SETTINGS):
    logpath = SETTINGS['recording_log']
    if not os.path.isfile(logpath):
        print 'no log file found at', logpath
        return []
    else:
        with open(logpath,'rb') as f:
            return f.read().split()




#-----------------------------------
# pre-processing of recorded samples
#-----------------------------------

def cutSeries(data, sampleSeries, SETTINGS, SUBJECT, dst, order = 7, lowcut = 100, highcut = 10000, 
              threshold = 0.1, loffset = 2400, roffset = 2400):
    

    print("dst", dst)


    """
    Method to automatically cut the 10 recorded digits into single digit recordings.
    Parameters:
    -----------
        data: numpy array
            Recorded audio data.
        sampleSeries: list 10 of (digit, repetition) tuples
            Expected labels of recordings.
        SETTINGS: dict
        SUBJECT: dict
        dst: string
            Destination where to store the cut wav files
        order: int
            order of bandpass filter
        lowcut: int
            lower limit of bandpass filter
        highcut: int
            higher limit of bandpass filter
    """

    success = True
    data = data.flatten()
    data = data[24000:-24000]
    fs = SETTINGS['samplerate']
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = scipy.signal.butter(order, [low, high], btype='band')
    #b, a = scipy.signal.cheby2(order, 40, [low, high], btype='bandpass')
    #y = scipy.signal.lfilter(b, a, data)
    y = scipy.signal.filtfilt(b, a, data)  
    y = y / np.percentile(abs(y), 99)
    rolledMean = pd.rolling_max(arg = abs(y), window=int(1*4800), center = True)
    rolledMean[np.isnan(rolledMean)] = 0
    idcs = np.where(rolledMean > 0.1)[0]
    stopIdcs = np.concatenate([idcs[np.where(np.diff(idcs) > 1)[0]], [idcs[-1]]])

    revIdcs = idcs[::-1]
    startIdcs = np.concatenate([[revIdcs[-1]], revIdcs[np.where(np.diff(revIdcs) < -1)[0]][::-1]])
   
    startIdcs_orig = startIdcs[:]
    stopIdcs_orig = stopIdcs[:]
    
    if not len(startIdcs) == len(stopIdcs):
        print("Problem cutting digit series: number of start and stop indices \
               should be equal but are {}, {}".format(len(startIdcs), len(stopIdcs)))
        success =  False

    elif np.any(stopIdcs - startIdcs > 48000):
        print("Found sample with duration > 1 sec")
        success = False

    elif len(startIdcs)>10:
        print("file {}: Found more than 10 possible samples. Attempting to correct selection.")
        absSums = []
        
        for start, stop in zip(startIdcs, stopIdcs):
            absSums.append(np.sum(abs(y[start:stop])))
            
        while len(startIdcs) > 10:
            discardIdx = np.argmin(absSums)
            d1 = startIdcs[discardIdx] - stopIdcs[discardIdx-1] 
            d2 = stopIdcs[discardIdx] - startIdcs[discardIdx]
            
            if d2 < 3.5 * 4800 and d1 < 1.5*4800 and discardIdx != 0:
                # combine two selections: important to include the "t" at the end of "eigh-t"
                startIdcs = startIdcs[np.arange(0,len(startIdcs)) != discardIdx]
                stopIdcs = stopIdcs[np.arange(0,len(stopIdcs)) != (discardIdx - 1)]
            else:    
                # discard a selection
                startIdcs = startIdcs[np.arange(0,len(startIdcs)) != discardIdx]
                stopIdcs = stopIdcs[np.arange(0,len(stopIdcs)) != discardIdx]
            absSums.pop(discardIdx)

    if len(startIdcs)<10:
        print("Found less than 10 digits in record.")
        success = False

    elif np.any(stopIdcs - startIdcs > 48000):
        print("Found sample with duration > 1 sec")
        success = False
    
    if not os.path.exists(dst):
        os.makedirs(dst)

    if success:
        for digitIdx, (start, stop) in enumerate(zip(startIdcs, stopIdcs)):
    
            d,r = sampleSeries[digitIdx]
            subj_name = SUBJECT["alias"]
            wavfile.write(os.path.join(dst, str(d) + '_' + subj_name + '_' + str(r) + '.wav'), 
                                48000, data[start-loffset: stop+roffset])
            
            print(os.path.join(dst, str(d) + '_' + subj_name + '_' + str(r) + '.wav'))

    fig, ax = plt.subplots(2,1,figsize = (20,5))

    # set below to True to visualize
    if False: #not success
            ax[0].plot(data, 'k')
            ax[0].plot(range(start-loffset,stop+roffset),data[start-loffset:stop+roffset])
            ax[0].plot(startIdcs_orig, np.zeros_like(startIdcs_orig), '.c', ms = 12)
            ax[0].plot(stopIdcs_orig, np.zeros_like(stopIdcs_orig), '.m', ms = 12)
            ax[1].plot(y, 'k')
            ax[1].plot(rolledMean, color = 'mediumvioletred')
            plt.suptitle(str(sampleSeries))
            ax[1].plot(range(start-loffset,stop+roffset),       y[start-loffset:stop+roffset])
    if False:#not success:
        figpath = wavpath.rstrip(".wav") + "_unableToCut.pdf"
        existCounter = 1
        while os.path.exists(figpath):
            figpath = figpath.rstrip(".pdf") + str(existCounter) + ".pdf"
            existCounter += 1
        plt.savefig(figpath)
        print("save figure at: {}".format(figpath))
    return success




