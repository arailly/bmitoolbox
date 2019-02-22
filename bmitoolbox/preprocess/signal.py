import numpy as np
from scipy.signal import butter, lfilter, firwin
import os, h5py, glob
import bmitoolbox as bt
import numpy as np
from scipy.io import loadmat
from scipy import signal

from tqdm import tqdm_notebook as tqdm_nb


def z_score(x, axis=0):
    mean = x.mean(axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    return (x - mean) / std


def biporar_diff(data, sfreq=2000, timedelta=0, normalize=True):
    n_biporars = 58
    chan_diffs = np.empty([len(data), n_biporars])

    # calc diff
    chan_diffs[:,  0] = data[:,  0] - data[:,  1]
    chan_diffs[:,  1] = data[:,  1] - data[:,  2]
    chan_diffs[:,  2] = data[:,  2] - data[:,  3]
    chan_diffs[:,  3] = data[:,  3] - data[:,  4]
    chan_diffs[:,  4] = data[:,  4] - data[:,  5]

    chan_diffs[:,  5] = data[:,  6] - data[:,  7]
    chan_diffs[:,  6] = data[:,  7] - data[:,  8]
    chan_diffs[:,  7] = data[:,  8] - data[:,  9]
    chan_diffs[:,  8] = data[:,  9] - data[:, 10]
    chan_diffs[:,  9] = data[:, 10] - data[:, 11]

    chan_diffs[:, 10] = data[:, 12] - data[:, 13]
    chan_diffs[:, 11] = data[:, 13] - data[:, 14]
    chan_diffs[:, 12] = data[:, 14] - data[:, 15]
    chan_diffs[:, 13] = data[:, 15] - data[:, 16]
    chan_diffs[:, 14] = data[:, 16] - data[:, 17]

    chan_diffs[:, 15] = data[:, 18] - data[:, 19]
    chan_diffs[:, 16] = data[:, 19] - data[:, 20]
    chan_diffs[:, 17] = data[:, 20] - data[:, 21]
    chan_diffs[:, 18] = data[:, 21] - data[:, 22]
    chan_diffs[:, 19] = data[:, 22] - data[:, 23]

    chan_diffs[:, 20] = data[:, 24] - data[:, 25]
    chan_diffs[:, 21] = data[:, 25] - data[:, 26]
    chan_diffs[:, 22] = data[:, 26] - data[:, 27]
    chan_diffs[:, 23] = data[:, 27] - data[:, 28]

    chan_diffs[:, 24] = data[:, 29] - data[:, 30]
    chan_diffs[:, 25] = data[:, 30] - data[:, 31]
    chan_diffs[:, 26] = data[:, 31] - data[:, 32]
    chan_diffs[:, 27] = data[:, 32] - data[:, 33]

    chan_diffs[:, 28] = data[:, 34] - data[:, 35]
    chan_diffs[:, 29] = data[:, 35] - data[:, 36]
    chan_diffs[:, 30] = data[:, 36] - data[:, 37]
    chan_diffs[:, 31] = data[:, 37] - data[:, 38]

    chan_diffs[:, 32] = data[:, 39] - data[:, 40]
    chan_diffs[:, 33] = data[:, 40] - data[:, 41]
    chan_diffs[:, 34] = data[:, 41] - data[:, 42]
    chan_diffs[:, 35] = data[:, 42] - data[:, 43]

    chan_diffs[:, 36] = data[:, 44] - data[:, 45]
    chan_diffs[:, 37] = data[:, 45] - data[:, 46]
    chan_diffs[:, 38] = data[:, 46] - data[:, 47]
    chan_diffs[:, 39] = data[:, 47] - data[:, 48]

    chan_diffs[:, 40] = data[:, 49] - data[:, 50]
    chan_diffs[:, 41] = data[:, 50] - data[:, 51]
    chan_diffs[:, 42] = data[:, 51] - data[:, 52]
    chan_diffs[:, 43] = data[:, 52] - data[:, 53]

    chan_diffs[:, 44] = data[:, 54] - data[:, 55]
    chan_diffs[:, 45] = data[:, 55] - data[:, 56]
    chan_diffs[:, 46] = data[:, 56] - data[:, 57]
    chan_diffs[:, 47] = data[:, 57] - data[:, 58]

    chan_diffs[:, 48] = data[:, 59] - data[:, 60]
    chan_diffs[:, 49] = data[:, 60] - data[:, 61]
    chan_diffs[:, 50] = data[:, 61] - data[:, 62]
    chan_diffs[:, 51] = data[:, 62] - data[:, 63]

    chan_diffs[:, 52] = data[:, 66] - data[:, 67]
    chan_diffs[:, 53] = data[:, 67] - data[:, 68]

    chan_diffs[:, 54] = data[:, 69] - data[:, 70]
    chan_diffs[:, 55] = data[:, 70] - data[:, 71]
    chan_diffs[:, 56] = data[:, 71] - data[:, 72]
    chan_diffs[:, 57] = data[:, 72] - data[:, 73]

    # normalize
    if normalize:
        chan_diffs = np.array([z_score(chan_diffs[:, i])
                               for i in range(n_biporars)]).T

    return chan_diffs


def epoching(raw, trig, size: int, offset: int = 0, standardize: bool = False):
    """Synchronize raw wave to trigger data and make epochs.

    Params:
        raw (array-like, shape(n_samples, n_channels)):
            Brain wave array
        trig (array-like, shape(n_triggers,)):
            Triggers to synchronize raw wave to image of video.
        size (int):
            Epoch size.
        offset (int, optional, default=0):
            Time shift.
        sfreq (int, optional, default=2000):
            Sampling freqency.
        standardize (bool):
            True if you want to standardize data

    Returns:
        epochs (numpy.array, shape(n_epochs, n_samples, n_channels)):
            Array of a second long epoch.

    """
    epochs = []
    for i, trig_idx in enumerate(trig):
        epoch_indices = range(trig_idx + offset,
                              trig_idx + offset + size)
        epochs.append(raw[epoch_indices])

    return np.array(epochs)


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Butterworth Bandpass filter
    params:
        data: array-like
    Ref:
        https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def bandpass_filter(data, lowcut, highcut, fs=2000, numtaps=255):
    """Simple FIR Bandpass filter
    """
    nyq = fs / 2

    # design filter
    fe1 = lowcut / nyq
    fe2 = highcut / nyq
    b = firwin(numtaps, (fe1, fe2), pass_zero=False)

    filtered = lfilter(b, 1, data)

    return filtered


def bandstop_filter(data, lowcut, highcut, fs=2000, numtaps=255):
    """Simple FIR Bandstop filter
    """
    nyq = fs / 2

    # design filter
    fe1 = lowcut / nyq
    fe2 = highcut / nyq
    b = firwin(numtaps, (fe1, fe2), pass_zero=True)

    filtered = lfilter(b, 1, data)

    return filtered


def var_preprocess(raw_files, trig_files, output_dir, standardize=True, augmentation=1):
    """preprocess for video-annotation-regression"""
    for raw_file, trig_file in zip(tqdm_nb(raw_files), trig_files):

        # load brainwave
        raw = np.array(h5py.File(raw_file, 'r')['ecog_dat'])
        n_channels = raw.shape[-1]

        # load trigger
        trig = loadmat(trig_file)['trigger_index'][0]

        # filter
        # pass 1 - 150Hz
        bandpassed = np.array([
            bt.bandpass_filter(raw[:, i], 1, 150)
            for i in range(n_channels)
        ]).T

        # eliminate 59 - 61Hz
        bandstopped = np.array([
            bt.bandstop_filter(bandpassed[:, i], 59, 61)
            for i in range(n_channels)
        ]).T

        # epoching
        epochs = bt.epoching(bandstopped, trig, size=1000)

        # standardize
        standard_epochs = []

        for epoch in epochs:
            standard_samples = [bt.z_score(epoch[:, i]) for i in range(n_channels)]
            standard_epochs.append(np.array(standard_samples).T)

        standard_epochs = np.array(standard_epochs)

        print(standard_epochs.shape)

        filename = raw_file.split('/')[-1].split('.')[0]
        bt.write_pickle(standard_epochs, output_dir+filename)
