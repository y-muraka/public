import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

if __name__ == "__main__":
    fs = 400e3
    T = 105e-3

    Traise = 5e-3

    tscale = np.linspace(0, T, int(fs*T))
    tscale_raise = np.linspace(0, Traise, int(fs*Traise))

    f_input = np.arange(100,8110,10)

    window_raise = np.append(np.sin(np.pi/2*tscale_raise/Traise), np.ones(int(round((T-Traise)*fs))))

    for ii in range(f_input.size):
        w = 2*np.pi*f_input[ii]
        x = np.sin(w*tscale) * window_raise

        np.save('./stim/%dHz'%(f_input[ii]), x)