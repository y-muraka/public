import numpy as np
import CochlearModel_SSAM2019_2D_semiFFT as am
import stimfile
import os

if __name__ == "__main__":

    Nx = 1500
    Ny = 20

    mode = '2D'
    fp = np.array([990, 1410, 2810, 3480, 3730, 3990, 4010, 4050, 4100, 4150, 4290, 4590, 4920, 5280, 6010, 8010])
    Lp = np.arange(20,90,10)
    
    
    cm = am.CochlearModel(Nx, Ny)
    for ii in range(fp.size):
        for jj in range(Lp.size):
            filename_wav = './stim/%dHz.npy'%(fp[ii])
            dirname_data = 'data/cochlea2D/PureTone/%dHz/%ddB/'%(fp[ii], Lp[jj])
            print("%s %ddB"%(filename_wav, Lp[jj]))

            if os.path.exists(dirname_data) and os.path.exists(dirname_data+'vb.npy'):
                continue
            elif os.path.exists(dirname_data) == False:
                os.makedirs(dirname_data)

            stim = stimfile.load(filename_wav, Lp[jj], mode) # Loading input signal

            vb, ub, p = cm.solve_time_domain(stim)

            np.save(dirname_data+'vb', vb)