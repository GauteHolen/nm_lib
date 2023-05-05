import numpy as np


def init_sod_test(nump, rho_L,rho_R,Pg_L,Pg_R):
    xx = np.linspace(0,1,nump+1)
    u = np.zeros(nump+1)
    rho = np.ones(nump+1)*rho_L
    rho[int(nump/2):] = rho_R
    Pg = np.ones(nump+1)*Pg_L
    Pg[int(nump/2):] = Pg_R

    return xx,u,rho,Pg