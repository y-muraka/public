import numpy as np
import scipy.io as sio
from scipy.signal import resample
import matplotlib.pyplot as plt

dt = 5e-6
fs = int(np.round(1/dt))
filename_in = 'wav/original/1000Hz.wav'
filename = 'wav/converted/1000Hz.wav'

fs_in, xin = sio.wavfile.read(filename_in)
xin = xin.astype(np.float32)

num_x = int(np.round(xin.size * fs/fs_in))
x = resample(xin, num_x)
x = x/np.max(np.abs(x))

sio.wavfile.write(filename, fs, x)