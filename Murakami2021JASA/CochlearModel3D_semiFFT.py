import numpy as np
import matplotlib.pyplot as plt
import tqdm
from scipy.fft import dct, idct
from time import time

class params_cochlea:
    def __init__(self, Nx, Ny, Nz, gamma):
        chdamp = 2
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.Lb = 2.5
        self.H = 0.1
        self.W = 0.1
        self.rho = 1.0
        self.dx = self.Lb/Nx
        self.dy = self.H/(Ny-1)
        self.dz = self.W/(Nz-1)
        self.x = np.arange(0, self.Lb, self.dx)
        self.y = np.linspace(0, self.H, Ny)
        self.z = np.linspace(0, self.W, Nz)

        self.xx, _ = np.meshgrid(self.x, self.z)
        self.xx = self.xx.T
        self.k1 = 1.1e9*np.exp(-4*self.xx)
        self.m1 = 3e-3
        self.c1 = 20 + 1500*np.exp(-2*self.xx) * chdamp
        self.k2 = 7.0e6 * np.exp(-4.4*self.xx)
        self.c2 = 10*np.exp(-2.2*self.xx) * chdamp
        self.m2 = 0.5e-3
        self.k3 = 1e7*np.exp(-4*self.xx)
        self.c3 = 2.0*np.exp(-0.8*self.xx) * chdamp
        self.k4 = 6.15e8*np.exp(-4*self.xx)
        self.c4 = 1040*np.exp(-2*self.xx) * chdamp
        self.g = 1
        self.b = 0.4
        self.gamma = gamma

        self.c1c3 = self.c1 + self.c3
        self.k1k3 = self.k1 + self.k3
        self.c2c3 = self.c2 + self.c3
        self.k2k3 = self.k2 + self.k3

        self.dt = 2e-6#5e-6
        self.beta = 50e-7

def Gohc(uc, beta):
    return beta*np.tanh(uc/beta)

def dGohc(uc, vc, beta):
    return vc/np.cosh(uc)**2

def get_g(pc, vb, ub, vt, ut):

    gb = pc.c1c3*vb + pc.k1k3*ub - pc.c3*vt - pc.k3*ut
    gt = - pc.c3*vb - pc.k3*ub + pc.c2c3*vt + pc.k2k3*ut

    uc_lin = ub - ut
    vc_lin = vb - vt

    uc = Gohc(uc_lin, pc.beta)
    vc = dGohc(uc_lin, vc_lin, pc.beta)

    gb -= pc.gamma * ( pc.c4*vc + pc.k4*uc )

    return gb, gt

#if __name__ == '__main__':
def solve_time_domain(Nx, Ny, Nz, gamma, f):
    Ntime = int(round(f.size/2))

    pc = params_cochlea(Nx, Ny, Nz, gamma)
    dt = pc.dt

    Tpre_start = time()

    alpha2 = 4*pc.rho*pc.b/pc.dy/pc.m1

    kx = np.arange(1,Nx+1)
    ax = np.pi*(2*kx-1)/4/Nx
    mwx = -4*np.sin(ax)**2/pc.dx**2

    kz = np.arange(Nz)
    az = np.pi * kz / 2 / (Nz - 1)
    mwz = -4*np.sin(az)**2/pc.dz**2

    mw = np.zeros((Nx,Nz))
    for mm in range(Nx):
        for kk in range(Nz):
            mw[mm,kk] = mwx[mm] + mwz[kk]

    vb = np.zeros((Ntime,Nx,Nz))
    ub = np.zeros((Ntime,Nx,Nz))
    vt = np.zeros((Ntime,Nx,Nz))
    ut = np.zeros((Ntime,Nx,Nz))

    p = np.zeros((Ntime,Nx,Nz))

    Ay = np.zeros((Nx,Nz,Ny,Ny))
    for mm in range(Nx):
        for kk in range(Nz):
            Ay[mm,kk,0,0] = -2 + mw[mm,kk]*pc.dy**2 - alpha2*pc.dy**2
            Ay[mm,kk,0,1] = 2
            for nn in range(1,Ny-1):
                Ay[mm, kk, nn, nn-1] = 1
                Ay[mm, kk, nn, nn] = -2 + mw[mm,kk]*pc.dy**2
                Ay[mm, kk, nn, nn+1] = 1
            Ay[mm, kk, Ny-1,Ny-2] = 2
            Ay[mm, kk, Ny-1,Ny-1] = -2 + mw[mm,kk]*pc.dy**2
    Ay /= pc.dy**2
    
    phat = np.zeros((Nx,Nz,Ny))
    iAy = np.zeros((Nx,Nz,Ny,Ny))
    for mm in range(Nx):
        for kk in range(Nz):
            iAy[mm,kk] = np.linalg.inv(Ay[mm,kk])

    Tpre = time() - Tpre_start

    Tmain_start = time()

    k = np.zeros((Nx,Nz,Ny))
    for ii in tqdm.tqdm(range(Ntime-1)):
        ######### RK4 ##################

        # (ii)
        gb, gt = get_g(pc, vb[ii], ub[ii], vt[ii], ut[ii])

        k[:,:,0] = -alpha2*gb
        k[0] = -f[ii*2] * 2/pc.dx
        
        #(iii)
        khat_xz = dct(dct(k, axis=0, type=3),axis=1, type=1)
        
        for mm in range(Nx):
            for kk in range(Nz):
                phat[mm,kk] = np.dot(iAy[mm,kk], khat_xz[mm,kk])
        
        p[ii] = idct(idct(phat, axis=0, type=3),axis=1, type=1)[:,:,0]

        #(iv)-(v)
        dvb1 = (p[ii]-gb)/pc.m1 
        ub1 = ub[ii] + 0.5*dt*vb[ii]
        vb1 = vb[ii] + 0.5*dt*dvb1

        dvt1 = -gt/pc.m2
        ut1 = ut[ii] + 0.5*dt*vt[ii]
        vt1 = vt[ii] + 0.5*dt*dvt1    

        # (ii)
        gb, gt = get_g(pc, vb1, ub1, vt1, ut1) 

        k[:,:,0] = -alpha2*gb
        k[0] = -f[ii*2+1] * 2/pc.dx

        #(iii)
        khat_xz = dct(dct(k, axis=0, type=3),axis=1, type=1)
        
        for mm in range(Nx):
            for kk in range(Nz):
                phat[mm,kk] = np.dot(iAy[mm,kk], khat_xz[mm,kk])
        
        p1 = idct(idct(phat, axis=0, type=3),axis=1, type=1)[:,:,0]

        #(iv)-(v)
        dvb2 = (p1-gb)/pc.m1
        ub2 = ub[ii] + 0.5*dt*vb1
        vb2 = vb[ii] + 0.5*dt*dvb2

        dvt2 = -gt/pc.m2
        ut2 = ut[ii] + 0.5*dt*vt1
        vt2 = vt[ii] + 0.5*dt*dvt2   

        # (ii)
        gb, gt = get_g(pc, vb2, ub2, vt2, ut2)

        k[:,:,0] = -alpha2*gb
        k[0] = -f[ii*2+1] * 2/pc.dx

        #(iii)

        khat_xz = dct(dct(k, axis=0, type=3),axis=1, type=1)
        
        for mm in range(Nx):
            for kk in range(Nz):
                phat[mm,kk] = np.dot(iAy[mm,kk], khat_xz[mm,kk])
        
        p2 = idct(idct(phat, axis=0, type=3),axis=1, type=1)[:,:,0]

        #(iv)-(v)
        dvb3 = (p2-gb)/pc.m1
        ub3 = ub[ii] + dt*vb2 
        vb3 = vb[ii] + dt*dvb3

        dvt3 = -gt/pc.m2
        ut3 = ut[ii] + dt*vt2
        vt3 = vt[ii] + dt*dvt3  

        # (ii)
        gb, gt = get_g(pc, vb3, ub3, vt3, ut3)
        
        k[:,:,0] = -alpha2*gb
        k[0] = -f[ii*2+2] * 2/pc.dx

        #(iii)

        khat_xz = dct(dct(k, axis=0, type=3),axis=1, type=1)
        
        for mm in range(Nx):
            for kk in range(Nz):
                phat[mm,kk] = np.dot(iAy[mm,kk], khat_xz[mm,kk])
        
        p3 = idct(idct(phat, axis=0, type=3),axis=1, type=1)[:,:,0]

        #(iv)-(v)
        dvb4 = (p3-gb)/pc.m1

        dvt4 = -gt/pc.m2  

        ub[ii+1] = ub[ii] + dt/6*(vb[ii] + 2*vb1 + 2*vb2 + vb3)
        vb[ii+1] = vb[ii] + dt/6*(dvb1 + 2*dvb2 + 2*dvb3 + dvb4) 
        ut[ii+1] = ut[ii] + dt/6*(vt[ii] + 2*vt1 + 2*vt2 + vt3)
        vt[ii+1] = vt[ii] + dt/6*(dvt1 + 2*dvt2 + 2*dvt3 + dvt4)

    Tmain = time() - Tmain_start
    return vb, ub, p, Tpre, Tmain

if __name__ == "__main__":
    T = 4e-3

    Nx = 400
    Ny = 8
    Nz = 8

    fp = 4000
    w = 2*np.pi*fp

    Lp = 60

    g = 0.75
    gamma = np.ones((Nx,Nz)) * g
    gamma[:,:2] = 0
    gamma[:,6:] = 0
    pc = params_cochlea(Nx, Ny, Nz, gamma)

    tscale = np.arange(0,T,pc.dt)
    tscale2 = np.arange(0,T,pc.dt/2)

    Ap = 20e-6 * 10**(Lp/20)

    sinewave = Ap*w*np.sin(w*tscale2)

        
    vb, ub, p, Tpre, Tmain = solve_time_domain( Nx, Ny, Nz, gamma, sinewave)

    np.save('tmp/vb', vb)