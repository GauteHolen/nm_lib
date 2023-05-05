from re import T
import numpy as np
from .hydro import step_density,step_momentum,step_pressure,step_energy,pad,unpad

class Species:

    k_b = 1.380649e-23

    def __init__(self,name,u0,rho0,Pg0,gamma,m=1,cD=0,cq=0,cL=0):
        self.name = name
        self.m = m
        self.u = u0
        self.rho = rho0
        self.Pg = Pg0
        self.gamma = gamma
        self.e = Pg0/(gamma-1) + 0.5*rho0*u0**2
        self.lenx = len(u0)
        self.cq = cq
        self.cL = cL
        self.cD = cD
        self.update_T()


    def init_store_values(self,Nt):
        self.us = np.zeros((Nt,self.lenx))
        self.rhos = np.zeros((Nt,self.lenx))
        self.Pgs = np.zeros((Nt,self.lenx))
        self.es = np.zeros((Nt,self.lenx))

        self.us[0] = self.u
        self.rhos[0] = self.rho
        self.Pgs[0] = self.Pg
        self.es[0] = self.e
    
    def store_values(self,i):
        self.us[i] = self.u
        self.rhos[i] = self.rho
        self.Pgs[i] = self.Pg
        self.es[i] = self.e
        self.update_T()

    def get_values(self):
        return [self.name,self.us,self.rhos,self.Pgs,self.es]

    def update_T(self):
        self.T = np.abs(self.m*self.u**2)/(self.k_b*3)


class Multispecies:

    k_b = 1.380649e-23

    def __init__(self,xx,ddx,bnd_limits,bnd_type):
        self.xx = xx
        self.dx = xx[1]-xx[0]
        self.species = []
        self.ddx = ddx
        self.bnd_limits = bnd_limits
        self.bnd_type = bnd_type
    
    def add_species(self,name,u0,rho0,Pg0,gamma, m = 1, cq = 0, cL =0, cD = 0):
        new_species = Species(name,u0,rho0,Pg0,gamma, m = m, cq = cq, cL = cL, cD = cD)
        self.species.append(new_species)

    def run(self,Nt,cfl_cut = 0.98,t0=0, coupled = False, Q_col = True, col = True):
        self.Q_col = Q_col
        self.col = col
        self.coupled = coupled
        self.xx = np.pad(self.xx,self.bnd_limits,"reflect", reflect_type='odd')

        self.t = t0
        self.cfl_cut = cfl_cut

        #Set up arrays for storing values:
        for s in self.species:
            s.init_store_values(Nt)

        
        tt = np.zeros(Nt)
        tt[0] = t0
        

        for i in range(1,Nt):
            self.update()
            for s in self.species:
                s.store_values(i)
            tt[i] = self.t

        data_array = []
        for s in self.species:
            data_array.append(s.get_values())

        self.xx = self.xx[self.bnd_limits[0]:-self.bnd_limits[1]]

        return tt,self.xx,data_array


    def get_dt(self):
        dt = np.inf
        for s in self.species:
            if np.amin(s.Pg) < 0:
                continue
            else:
                c = np.sqrt(s.gamma*s.Pg/s.rho) #Can't be zero because pressure/density can't be zero
                #c *= 0.9
                dtc = np.amin(self.dx/c)
                if np.amin(np.abs(s.u)) == 0:
                    dt_s = dtc
                else:
                    dtu = np.amin(self.dx/np.abs(s.u))
                    dt_s = min(dtu,dtc)
                dt_s = dt_s*self.cfl_cut
                if dt_s < dt:
                    dt = dt_s
        return dt



    




    def update(self):

        #Find common dt:
        
        dt = self.get_dt()
        self.t += dt

        for s in self.species:


            col = 0
            Q_col = 0
            if self.coupled:
                for sb in self.species:
                    if self.col:
                        col += self.get_col(s,sb)
                    if self.Q_col:
                        Q_col += self.get_Q_col(s,sb)

            [u,e,rho,Pg,col,Q_col] = pad([s.u,s.e,s.rho,s.Pg,col,Q_col],self.bnd_limits,self.bnd_type)

            gamma = s.gamma

            c = np.sqrt(gamma*Pg/rho) #Can't be zero because pressure/density can't be zero

            
            u = step_momentum(self.xx,u,rho,Pg,dt,cq = s.cq, cL = s.cL, col=col,ddx=self.ddx) / rho
            
            for j in range(len(u)):
                    if np.abs(u[j]) > np.abs(c[j]) and u[j] > 0:
                        u[j] = c[j]
                    elif np.abs(u[j]) > np.abs(c[j]) and u[j] < 0:
                        u[j] = -c[j]

            e = step_energy(self.xx,e,u,Pg,rho,dt,cq = s.cq, cL = s.cL, Q_col = Q_col,ddx=self.ddx)
            rho = step_density(self.xx,u,rho,dt,cD = s.cD, ddx=self.ddx)
            Pg = step_pressure(Pg,e,rho,u,gamma)


            [s.u,s.e,s.rho,s.Pg] = unpad([u,e,rho,Pg],self.bnd_limits) 

    
    def get_col(self, sa, sb):

        return sa.rho * self.get_col_freq(sa,sb)*(sa.u-sb.u)


    def get_col_freq(self, sa,sb):

        T_ab = (sa.m*sa.T + sb.m*sb.T) / (sa.m+sb.m)
        m_ab = (sa.m*sb.m)/(sa.m+sb.m)
        
        mu_a = np.zeros(len(sa.T))
        mu_b = np.zeros(len(sa.T))
        mu_ab = np.zeros(len(sa.T))
        uth_ab = np.zeros(len(sa.T))
        for i in range(len(sa.T)):
            if sa.T[i] > 1e-9 and sb.T[i] > 1e-9:
                mu_a[i] = sa.m/(self.k_b*sa.T[i])
                mu_b[i] = sa.m/(self.k_b*sb.T[i])
                mu_ab[i] = mu_a[i]*mu_b[i] / (mu_a[i]+mu_b[i])
                uth_ab[i] = np.sqrt(8/(np.pi*np.abs(mu_ab[i])))

        n_b = sb.rho/sb.m
        sigma = self.get_sigma_T(T_ab)

        return (m_ab/sa.m) * n_b * uth_ab * sigma


    def get_sigma_T(self,T_ab):
        return 1
    
    def get_Q_col(self, sa, sb):
        v_ab = self.get_col_freq(sa,sb)
        n_a = sa.rho/sa.m
        term1 = (n_a*sa.m*v_ab)/(sa.m+sb.m)
        term2 = sb.m * (sb.u-sa.u)**2
        term3 = 2*self.k_b*(sb.T-sa.T)/(sa.gamma-1)
        return term1 * (term2+term3)

    


    def update_uncoupled(self):

        #Find common dt:
        
        dt = self.get_dt()
        self.t += dt

        for s in self.species:

            [u,e,rho,Pg] = pad([s.u,s.e,s.rho,s.Pg],self.bnd_limits,self.bnd_type)

            gamma = s.gamma

            c = np.sqrt(gamma*Pg/rho) #Can't be zero because pressure/density can't be zero

            u = step_momentum(self.xx,u,rho,Pg,dt,cq = s.cq, cL = s.cL, ddx=self.ddx) / rho
            
            for j in range(len(u)):
                    if np.abs(u[j]) > np.abs(c[j]) and u[j] > 0:
                        u[j] = c[j]
                    elif np.abs(u[j]) > np.abs(c[j]) and u[j] < 0:
                        u[j] = -c[j]

            e = step_energy(self.xx,e,u,Pg,rho,dt,cq = s.cq, cL = s.cL, ddx=self.ddx)
            rho = step_density(self.xx,u,rho,dt,ddx=self.ddx)
            Pg = step_pressure(Pg,e,rho,u,gamma)


            [s.u,s.e,s.rho,s.Pg] = unpad([u,e,rho,Pg],self.bnd_limits) 