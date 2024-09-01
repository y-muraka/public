import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


if __name__ == "__main__":

    vb = np.load("./tmp/vb.npy")

    vb_max = np.max(np.abs(vb))
    vb_min = -vb_max

    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

    ims = []

    x = np.linspace(0,25,400)
    y = np.linspace(0,1,8)

    X, Y = np.meshgrid(x,y)
    X = X.T; Y = Y.T

    N_skip = 10
    for ii in range(0, vb.shape[0], N_skip):
        im = ax.plot_surface(X, Y, vb[ii], alpha=0.3, color="tab:blue")
        text = ax.text(0, 0.5, vb_max*1.5, "%g [msec]"%(ii*3e-6*1000))
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_zlabel("BM vecolity [mm/s]")
        ims.append([im]+[text])                  # グラフを配列 ims に追加

    ani = animation.ArtistAnimation(fig, ims, interval=100)
    plt.show()