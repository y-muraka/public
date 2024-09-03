from tkinter import W
import matplotlib.pyplot as plt
import numpy as np
import os

def calc_STC_TC(mode):
   
    fp = 4000 
    Lp = 30

    T = 100e-3
    Tstart = 5e-3
    Tend = 105e-3

    dt = 5e-6
    fsamp = 1/dt

    nstart = int(round(fsamp*Tstart))
    nend = int(round(fsamp*Tend))
    
    df = int(round(1/T))
    fscale = np.arange(0,fsamp,df)

    nfpro = np.where(fscale == fp)[0][0]

    f_input = np.array([990, 1410, 2810, 3480, 3730, 3990, 4010, 4150, 4290, 4590, 4920, 5280, 6010, 8010])
    fsup = np.array([990, 1410, 2810, 3480, 3730, 3990, 4010, 4050, 4100, 4150, 4290, 4590, 4920, 5280, 6010, 8010])
    Lsup = np.arange(20,85,10)
    L_input = Lsup

    # Calc. maximum BM velocity
    dirname_input = 'data/cochlea%s/PureTone/%dHz/%ddB/'%(mode, fp, Lp)
    vbm_t = np.load(dirname_input + 'vb.npy')
    nloc = np.argmax(np.max(np.abs(vbm_t),axis=0))

    output_rms = np.zeros((f_input.size, L_input.size)) 
    for ii in range(f_input.size):
        for jj in range(L_input.size):
            dirname_input = 'data/cochlea%s/PureTone/%dHz/%ddB/'%(mode, f_input[ii], L_input[jj])
            vbm_t = np.load(dirname_input + 'vb.npy')

            output_rms[ii, jj] = (20*np.log10(np.sqrt(np.square(vbm_t[:,nloc]).mean(axis=0))))

    vsfoae_f = np.zeros((fsup.size, Lsup.size), dtype = complex)
    
    dirname_p = 'data/cochlea%s/PureTone/%dHz/%ddB/'%(mode,fp, Lp)
    vp_t = np.load(dirname_p + 'vb.npy')[nstart:nend,nloc]

    for ii in range(fsup.size):
        for jj in range(Lsup.size):
            dirname_s = 'data/cochlea%s/PureTone/%dHz/%ddB/'%(mode, fsup[ii], Lsup[jj])
            dirname_ps = 'data/cochlea%s/TwoTone/%dHz/%ddB/%dHz/%ddB/'%(mode, fp, Lp, fsup[ii], Lsup[jj])

            vs_t = np.load(dirname_s + 'vb.npy')[nstart:nend,nloc]
            vps_t = np.load(dirname_ps + 'vb.npy')[nstart:nend,nloc]

            vp_f = np.fft.fft(vp_t)/vp_t.size*2
            vs_f = np.fft.fft(vs_t)/vs_t.size*2
            vps_f = np.fft.fft(vps_t)/vps_t.size*2

            
            vsfoae_f[ii, jj] = vp_f[nfpro] + vs_f[nfpro] - vps_f[nfpro]
    
    return output_rms, vsfoae_f, f_input, L_input, fsup, Lsup

if __name__ == "__main__":

    fp = 4000
    Lp = 30 

    for mode in ['1D','2D']:

        if mode == '1D':
            lvf = [-98.5]
            lvs = [-11.7]
        else:
            lvf = [-93.3]
            lvs = [-6]

        output_rms, vsfoae_f, f_input, L_input, fsup, Lsup = calc_STC_TC(mode)    

        fig1 = plt.figure(figsize=(8,6))
        ax = fig1.add_subplot(111)

        cs1 = ax.contour(np.log2(f_input/fp), L_input, output_rms.T, colors = 'tab:orange', linestyles = 'solid', linewidths=3, levels=lvf) 
        cs2 = ax.contour(np.log2(fsup/fp), Lsup, 20*np.log10(np.abs(vsfoae_f.T)/np.max(np.abs(vsfoae_f))) , linewidths=3, colors = 'tab:blue', linestyles = 'solid', levels=lvs) 

        ax.vlines(0, np.min(Lsup), np.max(Lsup),linestyle='dotted')
        ax.hlines(Lp, np.log2(fsup[0]/fp), np.log2(fsup[-1]/fp),linestyle='dotted')
        ax.set_ylabel('Input level / Suppressor level [dB SPL]', fontsize=18)
        ax.set_xlabel('Input frequency (Oct. re.fp)', fontsize=18)
        
        ax.clabel(cs1, inline=True, fontsize=18, colors='tab:orange', fmt = 'FTC', manual = [(-1,50)])
        ax.clabel(cs2, inline=True, fontsize=18, colors='tab:blue', fmt = 'STC', manual = [(-1.25, 55)])

        plt.title('%s model'%(mode),fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.savefig('%s.pdf'%(mode))
    plt.show()

    # calc tip-frequency

    from scipy.interpolate import interp1d

    def QERB_calculation(TC,cfs,df):
        # Keffe et al 2008
        indx = np.argmax(TC)
        CF= cfs[indx]
        ERB= np.trapz(TC**2, dx = df)/np.max(np.abs(TC))**2
        Qerb= CF/ERB
        return Qerb


 
    def get_metrics(cs):
        x = []
        y = []
        for p in cs.get_paths():
            v = p.vertices
            x = np.append(x, v[:,0])
            y = np.append(y, v[:,1])

        f = interp1d(x, y)

        xnew = np.linspace(x[0],x[-1], num=100)
        xnew_lin = fp*2**xnew
        fnew = f(xnew)

        nfmin = np.argmin(fnew)
        
        fmin = np.min(fnew)
        filt = fmin-fnew

        dx_new_lin = xnew_lin[1] - xnew_lin[0]
        
        freq_tip = xnew[nfmin]


        Qerb = QERB_calculation(filt, xnew_lin, dx_new_lin)

        dx_lin_slope_high = np.log2(xnew_lin[-1]/xnew_lin[nfmin])
        slope_hf = (fnew[-1]-fnew[nfmin])/dx_lin_slope_high
        tip_to_tail = -fmin + fnew[0]

        #print(dx_slope_high)
        print('High frequency slope [dB/octave] %g'%(slope_hf))
        print('Tip-to-tail [dB] %g'%(tip_to_tail))
        print('Tip-frequency [Oct re fp] %g'%(freq_tip))
        print('Qerb %g'%(Qerb))

        return slope_hf, tip_to_tail, freq_tip, Qerb

    print('FTC-1D:')
    slope_hf, tip_to_tail, freq_tip, Qerb = get_metrics(cs1)
    print('STC-1D:')
    slope_hf, tip_to_tail, freq_tip, Qerb = get_metrics(cs2)