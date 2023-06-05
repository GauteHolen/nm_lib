import sodshock
import numpy as np

def ana_sod_shock(xx,gamma,tt, PgL,PgR,rhoL,rhoR,conserv=False):

    dustFrac = 0.0
    t = 0.2
    left_state = (PgL,rhoL,0)
    right_state = (PgR, rhoR, 0.)


    # left_state and right_state set pressure, density and u (velocity)
    # geometry sets left boundary on 0., right boundary on 1 and initial
    # position of the shock xi on 0.5
    # t is the time evolution for which positions and states in tube should be 
    # calculated
    # gamma denotes specific heat
    # note that gamma and npts are default parameters (1.4 and 500) in solve 
    # function
    nump = len(xx)
    Nt = len(tt)
    ut = np.zeros((Nt,nump))
    Pt = np.zeros((Nt,nump))
    rhot = np.zeros((Nt,nump))
    et = np.zeros((Nt,nump))
    for i,t in enumerate(tt,start=0):
        positions, regions, values = sodshock.solve(left_state=left_state, \
            right_state=right_state, geometry=(xx[0], xx[-1], 0.5), t=t, 
            gamma=gamma, npts=nump, dustFrac=dustFrac)
        # Printing positions
        Pt[i], rhot[i], ut[i] = values["p"], values["rho"], values["u"]
    if conserv:
        et = 0.5*ut**2 + Pt/((gamma-1)*rhot)
    else:
        et = 0.5*rhot*ut**2 + Pt/((gamma-1))
    
    return ["Analytical sod shock", ut,rhot,Pt,et]