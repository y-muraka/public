from tkinter import W
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from scipy.fft import dct, idct
from time import time

class CochlearModel:
    def __init__(self, Nx):
        self.Nx = Nx
        self.Lb = 35e-3
        self.H = 7e-3
        self.rho = 1000
        self.dx = self.Lb/Nx
        self.x = np.arange(0, self.Lb, self.dx)

        self.kw = 200

        self.m1 = 0.03
        self.m2 = 0.3*self.m1

        self.k1 = 1.1e10*np.exp(-2*self.kw*self.x)
        self.k2 = 0.055*(self.m2/self.m1)*self.k1
        self.k3 = 0.045*(self.m2/self.m1)*self.k1
        self.k4 = 0.02*(self.m2/self.m1)*self.k1

        self.c1 = 6500*np.exp(-self.kw*self.x)
        self.c3 = 0.3*self.c1
        self.c4 = 0.93*self.c1
        self.c2 = 2*self.m2*np.sqrt((self.k2+self.k3-self.k4)/self.m2) - self.c3 + self.c4

        self.c1c3 = self.c1 + self.c3
        self.k1k3 = self.k1 + self.k3
        self.c2c3 = self.c2 + self.c3
        self.k2k3 = self.k2 + self.k3

        self.xisat2 = (1.4e-8)**2
        self.dxisat2 = (1.75e-4)**2

        self.dt = 5e-6


    def get_g(self, vb, ub, vt, ut):

        gb = self.c1c3*vb + self.k1k3*ub - self.c3*vt - self.k3*ut
        gt = - self.c3*vb - self.k3*ub + self.c2c3*vt + self.k2k3*ut

        uc = ub - ut
        vc = vb - vt


        knl = self.k4*(1-np.tanh(uc**2/self.xisat2 + vc**2/self.dxisat2))
        cnl = self.c4*(1-np.tanh(uc**2/self.xisat2 + vc**2/self.dxisat2))

        gb -= cnl*vc + knl*uc
        gt += cnl*vc + knl*uc

        return gb, gt

    def solve_time_domain(self, f):
        dt = self.dt
        Nx = self.Nx
        
        Ntime = int(round(f.size/2))

        Tpre_start = time()

        alpha2 = 4*self.rho/self.H/self.m1

        kx = np.arange(1,Nx+1)
        ax = np.pi*(2*kx-1)/4/Nx
        mwx = -4*np.sin(ax)**2/self.dx**2

        vb = np.zeros((Ntime,Nx))
        ub = np.zeros((Ntime,Nx))
        vt = np.zeros((Ntime,Nx))
        ut = np.zeros((Ntime,Nx))

        p = np.zeros((Ntime,Nx))
        
        phat = np.zeros((Nx))

        Tpre = time() - Tpre_start

        Tmain_start = time()
        for ii in tqdm.tqdm(range(Ntime-1)):
            ######### RK4 ##################

            # (ii)
            gb, gt = self.get_g(vb[ii], ub[ii], vt[ii], ut[ii])

            k = -alpha2*gb
            k[0] -= f[ii*2] * 2/self.dx
            
            #(iii)
            khat = dct(k, type=3)
            phat = khat/(mwx-alpha2)
            p[ii] = idct(phat, type=3)

            #(iv)-(v)
            dvb1 = (p[ii]-gb)/self.m1 
            ub1 = ub[ii] + 0.5*dt*vb[ii]
            vb1 = vb[ii] + 0.5*dt*dvb1

            dvt1 = -gt/self.m2
            ut1 = ut[ii] + 0.5*dt*vt[ii]
            vt1 = vt[ii] + 0.5*dt*dvt1    
            
            # (ii)
            gb, gt = self.get_g(vb1, ub1, vt1, ut1) 

            k = -alpha2*gb
            k[0] -= f[ii*2+1] * 2/self.dx

            #(iii)

            khat = dct(k, type=3)
            phat = khat/(mwx-alpha2)
            p1 = idct(phat, type=3)

            #(iv)-(v)
            dvb2 = (p1-gb)/self.m1
            ub2 = ub[ii] + 0.5*dt*vb1
            vb2 = vb[ii] + 0.5*dt*dvb2

            dvt2 = -gt/self.m2
            ut2 = ut[ii] + 0.5*dt*vt1
            vt2 = vt[ii] + 0.5*dt*dvt2   

            # (ii)
            gb, gt = self.get_g(vb2, ub2, vt2, ut2)

            k = -alpha2*gb
            k[0] -= f[ii*2+1] * 2/self.dx

            #(iii)

            khat = dct(k, type=3)
            phat = khat/(mwx-alpha2)
            p2 = idct(phat, type=3)

            #(iv)-(v)
            dvb3 = (p2-gb)/self.m1
            ub3 = ub[ii] + dt*vb2 
            vb3 = vb[ii] + dt*dvb3

            dvt3 = -gt/self.m2
            ut3 = ut[ii] + dt*vt2
            vt3 = vt[ii] + dt*dvt3  

            # (ii)
            gb, gt = self.get_g(vb3, ub3, vt3, ut3)
            
            k = -alpha2*gb
            k[0] -= f[ii*2+2] * 2/self.dx

            #(iii)

            khat = dct(k, type=3)
            phat = khat/(mwx-alpha2)
            p3 = idct(phat, type=3)

            #(iv)-(v)
            dvb4 = (p3-gb)/self.m1

            dvt4 = -gt/self.m2  

            ub[ii+1] = ub[ii] + dt/6*(vb[ii] + 2*vb1 + 2*vb2 + vb3)
            vb[ii+1] = vb[ii] + dt/6*(dvb1 + 2*dvb2 + 2*dvb3 + dvb4) 
            ut[ii+1] = ut[ii] + dt/6*(vt[ii] + 2*vt1 + 2*vt2 + vt3)
            vt[ii+1] = vt[ii] + dt/6*(dvt1 + 2*dvt2 + 2*dvt3 + dvt4)


        Tmain = time() - Tmain_start
        return vb, ub, p

if __name__ == "__main__":
    T = 50e-3

    fp = 4000
    for Nx in [1500]:
        cm = CochlearModel(Nx)
        dt = cm.dt
        tscale = np.arange(0,T,dt)
        tscale2 = np.arange(0,T,dt/2)
        for Lp in range(0,20,20):
            w = 2*np.pi*fp
            Ap = 2e-6 * 10**(Lp/20)
            sinewave = Ap*w*np.sin(w*tscale2)
            
            vb, ub, p, Tpre, Tmain = cm.solve_time_domain(sinewave)

            #plt.plot(cm.x, 20*np.log10(np.max(np.abs(vb[int(round(tscale.size*9/10)):]), axis=0)))
            plt.semilogy(cm.x, np.max(np.abs(ub[int(round(tscale.size*9/10)):])*1e9, axis=0))
    plt.show()