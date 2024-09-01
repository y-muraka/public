import numpy as np
import matplotlib.pyplot as plt


def get_phasor(x, fp, dt, T):

    fs = 1/dt
    df = int(round(1/T))

    fscale = np.arange(0,fs,df)
    nfp = np.where(fscale >= fp)[0][0]

    xfp = np.fft.fft(x, axis=0)[nfp]/x.size

    return xfp

fp = 1000
dt = 10e-6
T = 0.1

for mode in ['fft','direct']:
    for N in [256, 512, 1024]:
        plt.figure()
        x = np.linspace(0, 35, N)
        for Lp in [0, 40, 80]:
            filename = './tmp/%s/mat_vb_%d_%g.dat'%(mode, N, Lp)
            v = np.fromfile(filename, dtype=np.float64)
            v = v.reshape((int(T/dt),N))

            vf = get_phasor(v, fp, dt, T)
            plt.plot(x, 20*np.log10(np.abs(vf)), lw = 2, label = '%d dB'%(Lp))
            plt.xlabel('$x$ [mm]', fontsize=14)
            plt.ylabel('$\dot{u}_1$ [dB]', fontsize=14)
            plt.tick_params(labelsize=12)

            plt.title("(%s, %d)"%(mode, N))
            plt.legend(fontsize=14)
            plt.tight_layout()

plt.show()
