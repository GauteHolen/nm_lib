import importlib
import numpy as np
from nm_lib import nm_lib as nm
import matplotlib.pyplot as plt

 
def test_deriv_upw():
    ddx = nm.deriv_upw
    a = -1
    plot=False
    for i in range(10):
        deriv_1sto(ddx,a,plot=False)

def test_deriv_dnw():
    ddx = nm.deriv_dnw
    a = -1
    plot=False
    for i in range(10):
        deriv_1sto(ddx,a,plot=False)

def deriv_1sto(ddx,a,plot=False):
    #Random initial values
    x0 = -2.6
    xf = 2.6
    nump = 16*2**np.random.randint(0,5)
    xx = np.linspace(x0,xf,nump+1)
    Nt = 10**np.random.randint(1,4)
    #print(nump,Nt)

    print(f"Testing ddx = {str(ddx)} | Nt = {Nt}, nump = {nump}")

    #Functions for analytical solution
    def u_x_t0(xx):
        return np.ma.cos(0.2*6*np.pi*xx)**2/np.cosh(5*xx*xx)
    
    def u_ana(xx,t,a):
        """Centered analytical solution"""
        dx = xx[1]-xx[0]
        padding = int(abs(t*a/dx))
        if a<0:
            x_pad = np.pad(xx, [padding,0],'reflect', reflect_type='odd')
            ana = u_x_t0(x_pad[:] - a*t)
            return ana[0:xx.shape[0]]
        elif a>0:
            x_pad = np.pad(xx, [0,padding],'reflect', reflect_type='odd')
            ana = u_x_t0(x_pad[:] - a*t)
            return ana[-xx.shape[0]:]

    u0 = u_x_t0(xx)
    tt, ut = nm.evolv_adv_burgers(xx,u0,Nt,a,keep_centered = a, bnd_limits = [0,1], ddx=nm.deriv_dnw)

    dx = xx[1]-xx[0]
    dt = tt[1]-tt[0]

    #print(dx,dt)
    
    #Analytical solution
    anat = np.zeros((Nt,xx.shape[0]))
    for i,t in enumerate(tt):
        anat[i] = u_ana(xx,t,a)


    #Error
    errt = np.zeros((Nt,xx.shape[0]))
    for i,t in enumerate(tt):
        errt[i] = np.abs(anat[i]-ut[i])

    max_err = [np.amax(err) for err in errt]
    mean_err = [np.mean(err) for err in errt]

    
    #Coefficients for error margin function
    coeff = [2*dx, 4.60205674]
    #print(coeff)
    func = np.polyval(coeff,tt/dt)

    
    if plot:
        plt.plot(tt/dt,max_err/(dx), label = "max error")
        plt.plot(tt/dt,mean_err/(dx), label = "mean error")
        plt.plot(tt/dt,func, label="tolerance")
        plt.legend()
        plt.xlabel("Timestep")
        plt.ylabel("error/dx")
    
    print(max_err)
    
    for fit, err in zip(func,max_err):
        assert fit > err


