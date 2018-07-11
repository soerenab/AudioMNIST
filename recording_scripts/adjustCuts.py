import numpy as np
from scipy.io import wavfile
import os
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal
import glob
from matplotlib.lines import Line2D
import scipy.spatial.distance
import argparse

class DragHandler(object):
    """ A simple class to handle Drag n Drop.

    This is a simple example, which works for Text objects only
    """
    def __init__(self, figure=None) :
        """ Create a new drag handler and connect it to the figure's event system.
        If the figure handler is not given, the current figure is used instead
        """

        if figure is None : figure = plt.gcf()
        # simple attibute to store the dragged text object
        self.dragged = None

        # Connect events and callbacks
        figure.canvas.mpl_connect("pick_event", self.on_pick_event)
        figure.canvas.mpl_connect("button_release_event", self.on_release_event)

    def on_pick_event(self, event):
        " Store which text object was picked and were the pick event occurs."

        if isinstance(event.artist, Line2D):
            self.dragged = event.artist
            self.pick_pos = (event.mouseevent.xdata, event.mouseevent.ydata)

        return True

    def on_release_event(self, event):
        " Update text position and redraw"


        if self.dragged is not None :
            orig_dragged = np.copy(self.dragged.get_xydata())
            clickIdx = self.pos2ind(self.dragged, self.pick_pos)
            old_pos = self.dragged.get_xydata()[clickIdx]
            new_pos = (old_pos[1] + event.xdata - self.pick_pos[1], 0)

            orig_dragged[clickIdx] = np.array(new_pos)
            self.dragged.set_data(orig_dragged.T)
            
            global all_markers
            all_markers = self.dragged.get_xydata()[:,0]

            self.dragged = None
            plt.draw()

        return True


    def pos2ind(self, dragged, pick_pos):
        alldists = scipy.spatial.distance.cdist(dragged.get_xydata(), np.atleast_2d(pick_pos))
        return np.argmin(alldists)



def butter_bandpass(lowcut, highcut, fs, order=7):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = scipy.signal.butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=7):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = scipy.signal.filtfilt(b, a, data)
    return y



def run(src, dst):

    """
    Function to semi-automatically cut series of audio recordings into single 
    recordings. Automatically determined cut positions will be displayed and
    can be corrected via drag&drop. Close the visualization to apply the cuts.
    Important: cut data is still "raw", the bandpass filter and other means are
    only applied to find cut positions. Cut data will be saved in single files,
    the original audio series recordings will not be deleted.

    Parameters:
    -----------
        src: string
            Source directory containing the audio series recordings.
        dst: string
            Destination directory where to store cut files.
    """

    # hyperparameters for finding cut positions.
    lowcut = 100
    highcut = 10000
    threshold = 0.1
    loffset = 2400
    roffset = 2400
    expectedNum = 10

    if not os.path.exists(dst):
        os.makedirs(dst)

    filenames = glob.glob(os.path.join(src, "*.wav"))

    for fileIdx, filename in enumerate(filenames):

        # infer digits and repetition from file name convention
        digitSeries = filename.rstrip('.wav').split('__')[1:]
        # load sampling frequency and recorded data
        thisFs, thisData = wavfile.read(filename)

        # find single articulations
        y = butter_bandpass_filter(thisData, lowcut, highcut, thisFs, order=7)
        y = y / np.percentile(abs(y), 99)

        rolledMean = pd.rolling_max(arg = abs(y), window=int(1*4800), center = True)
        rolledMean[np.isnan(rolledMean)] = 0
        idcs = np.where(rolledMean > 0.1)[0]
        stopIdcs = np.concatenate([idcs[np.where(np.diff(idcs) > 1)[0]], [idcs[-1]]])

        revIdcs = idcs[::-1]
        startIdcs = np.concatenate([[revIdcs[-1]], revIdcs[np.where(np.diff(revIdcs) < -1)[0]][::-1]])

        if np.any((stopIdcs - startIdcs) > 48000):
            print("Found sample with more than one second duration")

        assert(len(startIdcs) == len(stopIdcs))

        if len(startIdcs) < expectedNum:
            print("file {}: Found only {} candidate samples".format(fileIdx, len(startIdcs)))
            # appending artificial markers for drag&drop later on.
            tmp1 = np.arange(expectedNum)
            tmp1[0:len(startIdcs)] = startIdcs
            startIdcs = tmp1

            tmp2 = np.arange(expectedNum)
            tmp2[0:len(stopIdcs)] = stopIdcs
            stopIdcs = tmp2
            print("Corrected to {} startIdcs".format(len(startIdcs)))


        if len(startIdcs)>expectedNum:
            print("file {}: Found more than 10 possible samples. Attempting to correct selection.".format(fileIdx))
            
            # this is based on some experience, but does not always work
            absSums = []

            for start, stop in zip(startIdcs, stopIdcs):
                absSums.append(np.sum(abs(y[start:stop])))

            while len(startIdcs) > expectedNum:
                discardIdx = np.argmin(absSums)
                d1 = startIdcs[discardIdx] - stopIdcs[discardIdx-1] 
                d2 = stopIdcs[discardIdx] - startIdcs[discardIdx]
                if discardIdx >= 1:
                    newd = startIdcs[discardIdx - 1] - stopIdcs[discardIdx]
                else:
                    newd = None
                if d2 < 3.5 * 4800 and d1 < 1.5*4800 and discardIdx != 0:
                    # combine two selections: important to include the "t" at the end of "eigh-t"
                    startIdcs = startIdcs[np.arange(0,len(startIdcs)) != discardIdx]
                    stopIdcs = stopIdcs[np.arange(0,len(stopIdcs)) != (discardIdx - 1)]
                else:    
                    # discard a selection
                    startIdcs = startIdcs[np.arange(0,len(startIdcs)) != discardIdx]
                    stopIdcs = stopIdcs[np.arange(0,len(stopIdcs)) != discardIdx]
                absSums.pop(discardIdx)


        fig, ax = plt.subplots(2,1,figsize = (20,5))
        ax[0].plot(thisData, 'k')
        ax[1].plot(y, 'k')
        ax[1].plot(rolledMean, color = 'mediumvioletred')


        for digitIdx, (start, stop) in enumerate(zip(startIdcs, stopIdcs)):
            # plot single digit recording according to current markers
            d,r = digitSeries[digitIdx].split('_')
            ax[0].plot(range(start-loffset,stop+roffset),thisData[start-loffset:stop+roffset])
            ax[1].plot(range(start-loffset,stop+roffset),       y[start-loffset:stop+roffset])
            ax[1].text(start + (stop-start)/2, 1.3, str(d), fontsize = 15)

            if digitIdx == expectedNum-1:
                all_markers = np.zeros((startIdcs.size + stopIdcs.size))
                all_markers[0::2] = startIdcs - loffset
                all_markers[1::2] = stopIdcs + roffset
                ax[0].plot(all_markers, np.zeros_like(all_markers), '.', ms = 10, picker = 10, c = 'indigo')

        ax[0].set_xlim([0, len(thisData)])
        ax[1].set_xlim([0, len(thisData)])
        ax[0].set_title("{}, len = {}".format(digitSeries[:], len(thisData)))
        dragh = DragHandler()
        plt.show()

        all_markers = sorted(np.round(all_markers).astype(int))
        plt.figure(figsize = (20,10))
        
        for digitIdx, (markStart, markStop) in enumerate(zip(all_markers[0::2], all_markers[1::2])):
            # infer digit, repetition and subject identifier
            d, r = digitSeries[digitIdx].split('_')
            subj_name = os.path.split(filename)[-1].split("__")[0]
            # write out files
            print("writing to {}".format(os.path.join(dst, d + '_' + subj_name + '_' + r + '.wav')))
            wavfile.write(os.path.join(dst, d + '_' + subj_name + '_' + r + '.wav'), 48000, thisData[markStart: markStop])
            # visualize cut data
            plt.plot(range(markStart, markStop), thisData[markStart:markStop])
            
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-src', default=".", help='Source directory where recorded autio series are stored.')
    parser.add_argument('-dst', default="./cut", help='Destination directory where to store cut audio files.')
    args = parser.parse_args()

    run(src=args.src, dst=args.dst)


