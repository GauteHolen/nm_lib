from re import X
from .src import *

def evolve_hydro(xx,u0,rho0,nt,Pg0,gamma,cfl_cut = 0.98, bnd_limits = [2,2], bnd_type = "symmetric", ddx="Lax"):

    xx = np.pad(xx,bnd_limits,"reflect", reflect_type='odd')
    u = u0
    rho = rho0
    Pg = Pg0
    e0 = Pg/(gamma-1) + 0.5*rho0*u0**2
    e = e0

    es = np.zeros((nt,len(e0)))
    es[0] = e0 
    us = np.zeros((nt,len(u0)))
    us[0] = u0
    rhos = np.zeros((nt,len(rho0)))
    rhos[0] = rho0
    Ps = np.zeros((nt,len(Pg0)))
    Ps[0] = Pg0

    tt = np.zeros(nt)
    t = 0

    len_x = len(xx)
    
    dx = xx[1]-xx[0]    
    for i in range(1,nt):

        #print(f"rho = {rho} \t e = {e} \t u = {u}")

        #Padding:
        u = np.pad(u,bnd_limits,bnd_type)
        e = np.pad(e,bnd_limits,bnd_type)
        rho = np.pad(rho,bnd_limits,bnd_type)
        Pg = np.pad(Pg,bnd_limits,bnd_type)
        
        c = np.sqrt(gamma*Pg/rho) #Can't be zero because pressure/density can't be zero
        #c *= 0.9
        dtc = np.amin(dx/c)
        if np.amin(np.abs(u)) == 0:
            dt = dtc
        else:
            dtu = np.amin(dx/np.abs(u))
            dt = min(dtu,dtc)
        dt = dt*cfl_cut
        t+= dt

        #print(t,np.amax(c))


        


        if ddx == "Lax":
        #print(dt)z
            print("lax")
            mom = step_momentum(xx,u,rho,Pg,dt,ddx = deriv_dnw)# a = c)
            u = mom/rho
            rho = step_Lax(xx,-rho*u,dt)
            #u = boundaries(u, bnd_limits=bnd_limits)
            #u[-1] = u[0]
            if True:
                for j in range(len_x):
                    if np.abs(u[j]) > np.abs(c[j]) and u[j] > 0:
                        u[j] = c[j]
                    elif np.abs(u[j]) > np.abs(c[j]) and u[j] < 0:
                        u[j] = -c[j]

            e = step_Lax(xx,-u*(e+Pg),dt)

        else:
            """
            u = u - dt*ddx_boundaries(xx,-(rho*u**2 + Pg),ddx, bnd_limits = [2,2],bnd_type = "edge")/rho
            e = e - dt*ddx_boundaries(xx,-u*(e+Pg),ddx,bnd_limits = [2,2],bnd_type = "edge")
            rho = rho - dt*ddx_boundaries(xx,-rho*u,ddx,bnd_limits = [2,2],bnd_type = "edge")
            """
            
            u = step_momentum(xx,u,rho,Pg,dt,ddx=ddx) / rho
            
            for j in range(len_x):
                    if np.abs(u[j]) > np.abs(c[j]) and u[j] > 0:
                        u[j] = c[j]
                    elif np.abs(u[j]) > np.abs(c[j]) and u[j] < 0:
                        u[j] = -c[j]

            
            e = step_energy(xx,e,u,Pg,rho,dt,ddx=ddx)
            

            
            rho = step_density(xx,u,rho,dt,ddx=ddx)
            

        Pg = step_pressure(Pg,e,rho,u,gamma)
        #Pg[-1] = Pg[0]
        #e = rho*e + 0.5*rho*u**2

        u = u[bnd_limits[0]:-bnd_limits[1]]
        e = e[bnd_limits[0]:-bnd_limits[1]]
        rho = rho[bnd_limits[0]:-bnd_limits[1]]
        Pg = Pg[bnd_limits[0]:-bnd_limits[1]]

        es[i] = e
        us[i] = u
        rhos[i] = rho
        Ps[i] = Pg
        tt[i] = t

        """
        if np.amin(e) < 0:
            print("Unphysical behaviour")
            break
        if np.amin(rho) < 0:
            print("Unphysical behaviour")
            break
        if np.amin(Pg) < 0:
            print("Unphysical behaviour")
            break
        """
        

    return es,us,rhos,Ps,tt

def ddx_boundaries(xx,hh,ddx,bnd_limits = [2,2],bnd_type = "edge"):
    hh = np.pad(hh,bnd_limits,bnd_type)
    return ddx(xx,hh)[bnd_limits[0]:-bnd_limits[1]]

def step_pressure(Pg,e,rho,u,gamma):

    Pg_new =  (gamma-1)*(e-rho*u**2) #- rho*u**2
    #Pg_new =  (gamma-1)*(e)/rho

    return Pg_new


def get_dt(uts):
    return NotImplemented

def phi(r):
    min1 = min(r,(1+r)/2,1)
    min2 = min(2*r,(1+r)/2,2)
    return max(0,min1,min2)


"""
def step_flux_limiter(xx,hh,dt):

    r = (hh-np.roll(hh,-1)) / (np.roll(hh,1) + hh)


    f_phalf1 = 
    f_phalf2 =
    f_nhalf1 =
    f_nhalf2 =

    f_p = f_phalf1 + phi(r) * (f_phalf2 - f_phalf1)
    f_n = f_nhalf1 + phi(r) * (f_nhalf2 - f_nhalf1)


    hh_new = hh + dt*(f_n - f_p)

"""


def step_Lax(xx,hh,dt,a=1):
    if type(a) == np.ndarray:
        a = np.pad(a,[1,1],"edge")
    hh = np.pad(hh,[1,1],"edge")

    dx = xx[1]-xx[0]
    h_avg1 = np.roll(hh,1) + np.roll(hh,-1)
    h_avg2 = np.roll(hh,1) - np.roll(hh,-1)

    hh_new = 0.5*h_avg1 -dt/dx * a*0.5*h_avg2

    return hh_new[1:-1]




def step_momentum(xx,u,rho,Pg,dt,col = 0,cq = 0,cL = 0, ddx = lambda x,y: deriv_cent(x, y), bnd_type='edge', bnd_limits = [2,2]):
    """_summary_

    Args:
        xx (_type_): _description_
        u (_type_): _description_
        rho (_type_): _description_
        Pg (_type_): _description_
        dt (_type_): _description_
        y (deriv_cent): _description_
        ddx (_type_, optional): _description_. Defaults to lambdax.

    Returns:
        _type_: _description_
    """
    #ddx = deriv_Lax_Wendroff
    lmd = 1
    q = get_q_diffusive(xx,rho,u,cq, cL)

    mom = u*rho - dt*ddx(xx,rho*u*u) - dt*ddx(xx,Pg + q) + dt*col



    return mom



def step_density(xx,u,rho,dt, cD = 0, ddx = lambda x,y: deriv_cent(x, y)):
    """_summary_

    Args:
        xx (_type_): _description_
        u (_type_): _description_
        rho (_type_): _description_
        dt (_type_): _description_
        y (deriv_cent): _description_
        ddx (_type_, optional): _description_. Defaults to lambdax.

    Returns:
        _type_: _description
    """
    dx = xx[1]-xx[0]
    #art_diff = dx**2 * step_diff_burgers(xx,rho, a=u)
    Diff = cD*rho*dx

    rho_new = rho - dt*ddx(xx,rho*u) + dt*ddx(xx,Diff*ddx(xx,rho*u))
    #rho_new = rho - dt*ddx(xx,rho*u)


    return rho_new


def step_energy(xx,e,u,Pg,rho,dt, cq = 0, cL = 0, Q_col = 0, ddx = lambda x,y: deriv_cent(x, y)):
    """_summary_

    Args:
        xx (_type_): _description_
        e (_type_): _description_
        u (_type_): _description_
        Pg (_type_): _description_
        dt (_type_): _description_
        y (deriv_cent): _description_
        ddx (_type_, optional): _description_. Defaults to lambdax.

    Returns:
        _type_: _description_
    """
   
    q = get_q_diffusive(xx,rho,u,cq,cL, ddx = ddx)
    e_new = e - dt*ddx(xx,e*u) - dt*(Pg+q)*ddx(xx,u) + dt*Q_col

    return e_new


def get_q_diffusive(xx,rho,u,cq, cL,ddx = lambda x,y: deriv_cent(x, y)):

    "https://arxiv.org/pdf/2202.11084.pdf"


    dx = xx[1] - xx[0]
    deriv1 = ddx(xx,u)
    deriv2 = ddx(xx,deriv1)

    #deriv2 = step_diff_burgers(xx,u,1)
    q_RvN = cq*dx*dx*deriv2
    qLQ = cL*dx*ddx(xx,u)
    #for i in range(len(qLQ)):
    #    qLQ[i] = min(0,qLQ[i])
    return - rho * (q_RvN + qLQ)


def pad(arrs, bnd_limits, bnd_type):
    for i in range(len(arrs)):
        arrs[i] = np.pad(arrs[i],bnd_limits,bnd_type)
    return arrs

def unpad(arrs, bnd_limits):
    for i in range(len(arrs)):
        arrs[i] = arrs[i][bnd_limits[0]:-bnd_limits[1]]
    return arrs


