import numpy as np
import CochlearModel_SSAM2019_1D_semiFFT as am
import stimfile
import os
import matplotlib.pyplot as plt


if __name__ == "__main__":

    mode = '1D_semiFFT'
    Nx = 1500

    fp = 4000
    Lp = 30

    fsup = np.array([990, 1410, 2810, 3480, 3730, 3990, 4010, 4050, 4100, 4150, 4290, 4590, 4920, 5280, 6010, 8010])
    Lsup = np.arange(20,90,10)
    
    cm = am.CochlearModel(Nx)

    filename_probe = './stim/%dHz.npy'%(fp)
    probe = stimfile.load(filename_probe, Lp, mode)

    for ii in range(fsup.size):
        for jj in range(Lsup.size):
            filename_wav = './stim/%dHz.npy'%(fsup[ii])
            dirname_data = 'data/cochlea1D/TwoTone/%dHz/%ddB/%dHz/%ddB/'%(fp, Lp, fsup[ii], Lsup[jj])
            print("%s %ddB"%(filename_wav, Lsup[jj]))

            if os.path.exists(dirname_data):
                continue
            elif os.path.exists(dirname_data) == False:
                os.makedirs(dirname_data)

            suppressor = stimfile.load(filename_wav, Lsup[jj], mode)

            stim = probe + suppressor

            vb, ub, p = cm.solve_time_domain(stim)

            np.save(dirname_data+'vb', vb)