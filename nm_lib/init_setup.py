import numpy as np


def init_sod_test(nump, rho_L,rho_R,Pg_L,Pg_R):
    xx = np.linspace(0,1,nump+1)
    u = np.zeros(nump+1)
    rho = np.ones(nump+1)*rho_L
    rho[int(nump/2):] = rho_R
    Pg = np.ones(nump+1)*Pg_L
    Pg[int(nump/2):] = Pg_R

    return xx,u,rho,Pg


def init_vel_drift(nump,rho,Pg,T1,T2,m):
    k_b = 1.380649e-23

    xx = np.linspace(0,1e3,nump+1)
    #u1 = np.random.normal(u1,0.1,nump+1)
    #u2 = np.random.normal(u2,0.1,nump+1)

    u1 = np.sqrt(3*T1*k_b/m)
    u2 = np.sqrt(3*T2*k_b/m)

    u1 = np.ones(nump+1)*u1
    u2 = np.ones(nump+1)*u2
    
    rho = np.ones(nump+1)*rho
    Pg = np.ones(nump+1)*Pg

    return xx,rho,Pg,u1,u2