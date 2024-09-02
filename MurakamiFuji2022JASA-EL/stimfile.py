import numpy as np
import scipy.io.wavfile as sio
from scipy.signal import resample
import matplotlib.pyplot as plt

def load(filename, Lpeak, mode):
    """
    Solve the cochlear model in time domain

    Parameters
    ----------
    filename : string
        Wav file name 
    Lpeak : float
        Sound pressure level in input signal

    Returns:
    --------
    signal : ndarray
        Generated input signal for the cochlear model
    """

    fs = 400e3
    data = np.load(filename)


    dt = 1/fs
    T = data.size/fs

    numT = data.size

    rms = np.sqrt(np.mean(data**2))
    data = data/rms

    if mode == '2D' or mode == '1D_semiFFT':
        multi = 2e-6*10**(Lpeak/20.0)
    elif mode == '1D_direct':
        multi = 1e-3*10**(Lpeak/20.0)

    signal = np.zeros(data.shape)
    signal[1:-1] = multi*(data[2:]-data[:-2])/2/dt

    return signal
