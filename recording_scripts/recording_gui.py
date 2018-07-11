# -*- coding: utf-8 -*-
import sounddevice as sd
from psychopy import gui, visual, core, event
import numpy as np
import scipy.io.wavfile as wavfile
import datetime
import os
import sys

import util


class ProgressBar():
    
    """
    Class to visualize a progress bar on a psychopy window object
    """

    def __init__(self, window, fillup=0.0, msg= None, autoDraw = False, pos = [0.0, 0.0], 
                 relWidth = 0.9, relHeight = 0.05, lineColor = (1,1,1), 
                 fillColor = (1,1,1), backgoundColor = (-0.9, -0.9, -0.9)):
        
        """
        Parameters:
        -----------
            window: psychopy.visual.Window
            fillup: float
                Initial progress of progress bar, must be in [0,1].
            msg: string
                Message to show.
            autoDraw: boolean
                If True, progess bar is shown with every frame flip. If false,
                progress bar is not visible.
            pos: x,y tuple
                Coordinates of center of progress bar
            relWidth: float
                Width of progressbar relative to screen width
            relHeigth: float
                Height of progressbar relative to screen height
            lineColor: RGB tuple
                Color of progressbar contours
            fillColor: RGB tuple
                Color of filling of progressbar
            backgroundcolor: RGB tuple
                Color of non-filled progressbar

        """

        self.window = window
        self.relWidth = relWidth
        self.fillColor = fillColor
        self.x0 = pos[0] - self.relWidth
        self.x1 = pos[0] + self.relWidth
        self.y0 = pos[1] - relHeight
        self.y1 = pos[1] + relHeight

        # current progress
        x_progress = self.x0 + fillup*self.relWidth
        # "static" progress bar frame
        self.backBar = visual.ShapeStim(self.window, 
                                        pos = pos,
                                        vertices = [(self.x0, self.y0),  
                                                    (self.x0, self.y1), 
                                                    (self.x1, self.y1), 
                                                    (self.x1, self.y0)], 
                                        closeShape = True, 
                                        lineColor = lineColor, 
                                        fillColor = backgoundColor,
                                        autoDraw = autoDraw,
                                        units = 'norm')
        
        # "dynamic" bar of the progress bar, width and height are zero at init, 
        # the correct positions of all vertices are set in set_progress()
        self.progBar = visual.Rect(self.window, 
                                   pos = pos, 
                                   width = 0, 
                                   height = 0, 
                                   lineColor = lineColor, 
                                   fillColor = self.fillColor, 
                                   autoDraw = autoDraw, 
                                   units = 'norm')

        # percentage number which may be displayed additionally
        self.msgPrompt = visual.TextStim(self.window, 
                                         text = msg if msg != None else str(fillup), 
                                         pos = (pos[0], 2*self.y1), 
                                         height = 0.05, 
                                         units = 'norm')
        # there is no autoDraw parameter in TextStim constructor, so set manually
        self.msgPrompt.setAutoDraw(autoDraw) 
        

    def set_progress(self, fillup, msg=None):
        
        """
        Method to update current progress.
        Parameters:
        -----------
            fillup: float
                Current progress, must be in [0,1].
            msg: string (or None)
                If a string is passed, it will be displayed.

        """

        x_progress = self.x0 + fillup*(2*self.relWidth)
        self.progBar.setVertices([(self.x0, self.y0), 
                                  (self.x0, self.y1), 
                                  (x_progress, self.y1), 
                                  (x_progress, self.y0)])

        self.progBar.closeShape = True
        if msg != None:
            self.msgPrompt.setText(msg)
        else:
            self.msgPrompt.setText("{:.1f} %".format(fillup*100))


    def set_message(self, msg):
        
        """
        Method to display messages.
        Parameters:
        -----------
        msg: string
            Message to display.
        """

        self.msgPrompt.setText(msg)


    def set_AutoDraw(self, flag):
        
        """
        Method to toggle autodraw of progressbar.
        See documentation of psychopy for further explanation of the autodraw 
        functionality.
        Parameters:
        -----------
            flag: boolean
                Set autodraw to True or to False.
        """
        
        self.backBar.setAutoDraw(flag)
        self.progBar.setAutoDraw(flag)
        self.msgPrompt.setAutoDraw(flag)

        
    def set_color(self, color):
        
        """
        Method to set color of progess of progressbar.
        Parameters:
        -----------
            color: RGB tuple
        """
        
        self.progBar.setFillColor(color)



class SubjectMessageScreen():
    
    """
    Class to display a message dialog
    """

    def __init__(self, msg):
        
        """
        Parameters:
        -----------
            msg: string
                Message to show.
        """
        
        self.window = gui.Dlg(title='Message dialog')
        self.window.addText(msg)
        

    def show(self):
    
        """
        Call this method to show this SubjectMessageScreen.
        """
    
        self.window.show()
        if self.window.OK:
            return True
        else:
            return False



class SubjectDataScreen(gui.DlgFromDict):
    
    """
    Class for entering subject data.
    """
    
    def __init__(self, 
                dictionary,
                title='Subject information',
                order = ['alias','age','origin','accent','native speaker','gender','recordingdate'],
                fixed=['recordingdate']):

        """
        Parameters:
        -----------
            dictionary: dict
                Dictionary with keys 'alias','age','origin','accent',
                'native speaker','gender','recordingdate'
            title: string
                Window title
            order: list of strings
                List of keys, specifying their order on screen
            fixed: list of strings
                List of keys whose value is immutable on screen.
        """


        # call the constructor of the base class
        gui.DlgFromDict.__init__(self, 
                                dictionary = dictionary, 
                                title = title, 
                                order = order, 
                                fixed = fixed)


    # There is no show method as gui.DlgFromDict does not need one.




class DataRecordingScreen():

    """
    Class for audio data recording.
    """

    def __init__(self, SUBJECT, SETTINGS, SAMPLES, window=None, auto_cut=False):

        """
        Parameters:
        -----------
            SUBJECT: dict
            SETTINGS: dict
            SAMPLES: dict
            
            See main.py for examples of SUBJECT, SETTINGS and SAMPLES.
        """

        # define temporary variables
        displayWidth = 0.8
        frameRate = 60.
        secBetweenDigits = 0.

        # define object variables
        self.SUBJECT = SUBJECT
        self.SETTINGS = SETTINGS
        self.SAMPLES = SAMPLES
        self.nAtOnce = 10
        self.auto_cut = auto_cut


        assert (frameRate * SETTINGS['seconds']) % 1 == 0, \
                "Number of frames per digit must be a natural number"
        assert (frameRate * secBetweenDigits) % 1 == 0, \
                "Number of frames between digits must be a natural number"
        assert (frameRate * SETTINGS['pause']) % 1 == 0, \
                "Number of frames for break must be a natural number"
        assert ((self.nAtOnce * SETTINGS['seconds'] + \
                 SETTINGS['pause']) * SETTINGS['samplerate']) % 1 == 0, \
                "number of audio samples must be a natural number"


        # define color definitions
        self.col_background = (-0.7, -0.7, -0.7)
        self.col_inactive   = (0.4, 0.4, 0.5)
        self.col_active     = (0.0, 0.5, 0)
        self.col_pause      = (-0.1, -0.1, -0.1)
        self.col_rejected   = (-1, -1, -0.5)
        self.col_instructions=(0.3,0.3,0.3)
        self.col_testmessage= (0.6,0.6,0.)

        # define keys
        self.KEY_REJECT = 'return'
        self.KEY_REJECT_PREVIOUS = 'x'
        self.KEY_PAUSE = 'space'
        self.KEY_PLAYBACK = 'p'

        if type(window) == visual.window.Window:
            self.win = window
        else:
            self.win = visual.Window(color = self.col_background, 
                                    fullscr = True, 
                                    allowGUI = True, 
                                    units = 'norm',
                                    name = 'Spoken numbers recording',
                                    waitBlanking = False)

        # compute temporary varibles
        spacing = 2*displayWidth / (self.nAtOnce+2)

        # compute object variables
        self.framesPerDigit = int(frameRate * SETTINGS['seconds'])
        self.framesBetweenDigits = int(frameRate * secBetweenDigits)
        self.framesPause = int(frameRate * SETTINGS['pause'])
        self.digitStims = []

        self.controls = visual.TextStim(self.win, 
                                        text = '<SPACE>\tto (UN)PAUSE after sample\n<ENTER>\tto REJECT sample', 
                                        pos = (-0.5, -0.85), 
                                        height = 0.05, 
                                        color = self.col_instructions, 
                                        wrapWidth = 1000)
        self.controls.setAutoDraw(True)

        self.test_prompt = visual.TextStim(self.win, 
                                           text = 'TEST RUN', 
                                           pos = (0, 0.2), 
                                           height = 0.05, 
                                           color = self.col_testmessage)


        # session progress bar
        self.sessProg = ProgressBar(self.win, 
                               pos = (0, 0.3), 
                               autoDraw = True, 
                               backgoundColor = self.col_background,
                               fillColor = self.col_inactive,
                               lineColor = self.col_inactive)

        # progess bar for breaks between series of digits
        self.pauseProg = ProgressBar(self.win, 
                                pos = (0, -0.3), 
                                autoDraw = True,
                                backgoundColor = self.col_background,
                                fillColor = self.col_inactive,
                                lineColor = self.col_inactive)

        # create placeholders for digits
        for d in range(self.nAtOnce+2):
            self.digitStims.append(visual.TextStim(self.win, 
                                                   text = '>' if d <= 1 else '_', 
                                                   pos = (-displayWidth + spacing * d, 0),
                                                   height = 0.12, 
                                                   units = 'norm',
                                                   color = self.col_inactive))
            self.digitStims[d].setAutoDraw(False)


    def show(self, istest = False, noMic = False):
        
        """
        Method to display audio recording setup on screen. There will be 10
        digits presented as a series which should be read out loud when they
        turn green. The script will attempt to cut the recordings into 10 
        records, one per digit. If this fails, as for instance the break
        between to articulations is too short or there was too much noise,
        the recording will be saved with the suffix 'unableToCut' and the 
        series is repeated at the end of the experiment.

        Parameters:
            istest: boolean
                If True a single test run will be shown. No data will be saved
                for this run. If False, the entire audio recording will be 
                carried out.
            noMic: boolean
                Select True if no microphone is connected to the computer.
                Recordings will not be saved.


        """
        
        if istest:
            self.test_prompt.setAutoDraw(True)
        else:
            self.test_prompt.setAutoDraw(False)

        REJECT_TRIGGER = False 
        PAUSE_TRIGGER = False
        REJECT_PREVIOUS_TRIGGER = False

        self.sessProg.set_progress(0)
        self.sessProg.set_AutoDraw(True)
        self.pauseProg.set_progress(0, "Get Ready")
        self.pauseProg.set_AutoDraw(True)

        wavpath_previous = None
        saveRec = True
        leaveTest = False
        seriesIdx = 0
        numSeries = float(len(self.SAMPLES) / self.nAtOnce)
        nAudioSamples = int(((self.nAtOnce + 1) * self.SETTINGS["seconds"] + self.SETTINGS["pause"])) * self.SETTINGS["samplerate"]

        samplesToRepeat = []

        while (len(self.SAMPLES) > 0 and not leaveTest) or PAUSE_TRIGGER:

            if not PAUSE_TRIGGER:
                # get new set of samples to display
                if seriesIdx != 0 and not REJECT_TRIGGER:
                    samples_previous = currentSamples[:] # creates a copy
                currentSamples = self.SAMPLES[0:self.nAtOnce]
                self.SAMPLES = self.SAMPLES[self.nAtOnce:]

                # set digits on screen
                self.digitStims[0].setColor(self.col_active)
                for stimIdx, stim in enumerate(self.digitStims[2:]):
                    stim.setText(currentSamples[stimIdx][0])
                    # if first time, activate all stimuli 
                    if seriesIdx == 0:
                        stim.setAutoDraw(True)
                        if stimIdx == 0:
                            self.digitStims[0].setAutoDraw(True)
                            self.digitStims[1].setAutoDraw(True)

                # display pause progress bar
                for fIdx in range(self.framesPause):
                    self.pauseProg.set_progress(fillup = (fIdx+1)/float(self.framesPause), msg = "Get Ready")
                    self.win.flip()

                # write out record
                if seriesIdx != 0:
                    if noMic or istest or not saveRec:
                        print(("auto - " if istest or noMic else "") +"rejected:", wavpath)
                        seriesIdx -= 1
                        saveRec = True
                        REJECT_TRIGGER = False
                    else:
                        if self.auto_cut:
                            if not util.cutSeries(recording, samples_previous, self.SETTINGS, self.SUBJECT, os.path.join(self.SETTINGS["data_folder"], "cut")):
                                print("Series could not be cut")
                                samplesToRepeat.extend(samples_previous)
                                wavpath = wavpath.rstrip(".wav") + "_unableToCut.wav"
                            else:
                                print("Successfully cut.")

                        util.add_to_log(self.SETTINGS, wavpath)
                        print('writing',wavpath)
                        wavfile.write(wavpath, self.SETTINGS['samplerate'], recording)
                        wavpath_previous = wavpath

                # start new record
                recording = np.zeros((nAudioSamples, self.SETTINGS['channels']), dtype=np.int16)
                if not noMic:
                    sd.rec(nAudioSamples,
                        samplerate=self.SETTINGS['samplerate'],
                        channels=self.SETTINGS['channels'],
                        dtype=np.int16,
                        device=self.SETTINGS['device_index'],
                        out=recording)

                # deactivate first > symbol
                self.digitStims[0].setColor(self.col_inactive)

                # loop through digits
                for stimIdx, stim in enumerate(self.digitStims[1:]):
                    # paint digit in active color
                    stim.setColor(self.col_active)
                    # set progress (take only digits into account)
                    if stimIdx >= 1:
                        self.sessProg.set_progress(seriesIdx / (numSeries) + (stimIdx)/(numSeries * float(self.nAtOnce)))
                    # display digit for specified number of frames
                    for _ in range(self.framesPerDigit):
                        self.win.flip()
                    # paint digit in inactive color 
                    stim.setColor(self.col_inactive)
                
                    if len(event.getKeys(['return'])) > 0:
                        REJECT_TRIGGER = True
                        break

            pressedKeys = event.getKeys([self.KEY_PAUSE, self.KEY_REJECT, self.KEY_REJECT_PREVIOUS])

            if (self.KEY_REJECT in pressedKeys or REJECT_TRIGGER) and saveRec:
                REJECT_TRIGGER = True
                saveRec = False
                self.pauseProg.set_color(self.col_rejected)
                self.sessProg.set_progress(seriesIdx / numSeries)
                for fRejIdx in range(self.framesPause):
                    self.pauseProg.set_progress((fRejIdx+1)/float(self.framesPause), "REJECTED")
                    self.win.flip()
                self.pauseProg.set_color(self.col_inactive)
                self.SAMPLES.extend(currentSamples)

            if self.KEY_PAUSE in pressedKeys:
                PAUSE_TRIGGER = not PAUSE_TRIGGER

            if self.KEY_REJECT_PREVIOUS in pressedKeys:
                REJECT_PREVIOUS_TRIGGER = True

            
            if PAUSE_TRIGGER:
                if REJECT_TRIGGER:
                    self.pauseProg.set_color(self.col_rejected)
                    self.pauseProg.set_message("PAUSE - REJECTED")
                else:
                    self.pauseProg.set_color(self.col_pause)
                    self.pauseProg.set_message("PAUSE")
                self.win.flip()


            if not PAUSE_TRIGGER:
                if REJECT_PREVIOUS_TRIGGER:
                    fileToRemove = util.get_full_series_path(self.SETTINGS, self.SUBJECT, samples_previous)
                    if os.path.isfile(fileToRemove):
                        util.remove_recordings([fileToRemove], self.SETTINGS)
                    else:
                        print("Cannot remove previous: {} does not exist. \
                            \n Perhaps it has already been removed before?".format(fileToRemove))
                    self.SAMPLES.extend(samples_previous)
                    seriesIdx -= 1
                    self.sessProg.set_progress(seriesIdx/numSeries)
                    REJECT_PREVIOUS_TRIGGER = False
                    


                self.pauseProg.set_color(self.col_inactive)
                wavpath = util.get_full_series_path(self.SETTINGS, self.SUBJECT, currentSamples)
                # increase series counter
                seriesIdx += 1
                # leave test after one series of digits
                if istest:
                    self.SAMPLES.extend(currentSamples)
                    leaveTest = True

            if not REJECT_TRIGGER:
                samples_previous = currentSamples[:]


        for fIdx in range(self.framesPause):
            self.pauseProg.set_progress(fillup = (fIdx+1)/float(self.framesPause), msg = "Complete")
            self.win.flip()


        # write out last record
        if not istest and not noMic:# and not noMic:
            if self.auto_cut:
                if not util.cutSeries(recording, samples_previous, self.SETTINGS, self.SUBJECT, os.path.join(self.SETTINGS["data_folder"], "cut")):
                    print("Series could not be cut")
                    samplesToRepeat.extend(samples_previous)
                    wavpath = wavpath.rstrip(".wav") + "_unableToCut.wav"
                else:
                    print("Successfully cut.")

            util.add_to_log(self.SETTINGS, wavpath)
            print('writing',wavpath)
            wavfile.write(wavpath, self.SETTINGS['samplerate'], recording)
            wavpath_previous = wavpath

        return samplesToRepeat

    def close(self):
        self.win.close()