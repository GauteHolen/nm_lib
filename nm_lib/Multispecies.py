from cmath import pi
from re import I, T
from colisions import CrossSection
import numpy as np
from .hydro import step_density,step_momentum,step_momentum_NS,step_pressure,step_energy,pad,unpad

class Species:

    k_b = 1.380649e-23

    def __init__(self,name,u0,rho0,Pg0,gamma,m=1,cD=0,cq=0,cL=0, d=0, molecule = "h"):
        self.name = name
        self.m = m
        self.u = u0
        self.rho = rho0
        self.Pg = Pg0
        self.gamma = gamma
        self.e = Pg0/((gamma-1))+ 0.5*rho0*u0**2
        self.lenx = len(u0)
        self.cq = cq
        self.cL = cL
        self.cD = cD
        self.update_T()
        self.col = np.zeros(self.lenx)
        self.Q_col = np.zeros(self.lenx)
        self.sigma = np.pi*d**2
        self.molecule = molecule

        print(f"Adding species {name} of type {molecule}")




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
        #print("T = ",self.T[0])


class Multispecies:

    k_b = 1.380649e-23

    def __init__(self,xx,ddx,bnd_limits,bnd_type):
        self.xx = xx
        self.dx = xx[1]-xx[0]
        self.species = []
        self.ddx = ddx
        self.bnd_limits = bnd_limits
        self.bnd_type = bnd_type
    
    def add_species(self,name,u0,rho0,Pg0,gamma, m = 1, cq = 0, cL =0, cD = 0, d = 0, molecule = "h"):
        new_species = Species(name,u0,rho0,Pg0,gamma, m = m, cq = cq, cL = cL, cD = cD, d = d, molecule="h")
        self.species.append(new_species)

    def run(self,Nt,cfl_cut = 0.98,t0=0, cs_dir = "../nm_lib/nm_lib/cross-sections",coupled = False, Q_col = True, col = True, NS = False, conservative = False, flux_algo="LF"):
        self.Q_col = Q_col
        self.col = col
        self.coupled = coupled
        self.xx = np.pad(self.xx,self.bnd_limits,"reflect", reflect_type='odd')
        self.lenx = len(self.xx)
        self.t = t0
        self.cfl_cut = cfl_cut

        V_const = 1
        if coupled:
            self.V_colmat = self.init_V_col(V_const)


        #Set up arrays for storing values:
        for s in self.species:
            s.init_store_values(Nt)

        
        tt = np.zeros(Nt)
        tt[0] = t0
        
        if coupled:
            self.init_crosssections(cs_dir)
        
        

        if conservative:
            print(f"Running with scheme {flux_algo}")
            for s in self.species:
                s.e = s.Pg/((s.gamma-1)*s.rho)+ 0.5*s.rho*s.u**2
            if flux_algo == "ROE":
                self.flux = self.flux_Roe
                s.e = s.Pg/((s.gamma-1)*s.rho)+ 0.5*s.u**2
            else:
                self.flux = self.flux_Lax_Wendroff
            
            update = self.update_conservative
        else:
            print(f"Running with scheme {self.ddx.__name__}")
            update = self.update_diffusive

        print(f"Timestep 0 of {Nt}", end = "\r")
        for i in range(1,Nt):
            print(f"Timestep {i} of {Nt}", end = "\r")
            update()
            for s in self.species:
                s.store_values(i)
            tt[i] = self.t

        data_array = []
        for s in self.species:
            data_array.append(s.get_values())

        self.xx = self.xx[self.bnd_limits[0]:-self.bnd_limits[1]]

        return tt,self.xx,data_array


    def init_V_col(self,V_const):
        N = len(self.species)
        Nx = len(self.species[0].rho)
        V_mat = np.ones((N,N,Nx))
        for i in range(N):
            for j in range(N):
                if i == j:
                    V_mat[i][j][:] = 0
                elif j > i:
                    V_mat[i][j][:] = V_const
                elif j < i:
                    V_mat[i][j][:] = V_mat[j][i][:] * self.species[i].rho / self.species[j].rho
        return V_mat


    def update_V_col_const(self):
        N = len(self.species)
        for i in range(N):
            for j in range(N):
                if j < i:
                    self.V_colmat[i][j][:] = self.V_colmat[j][i][:] * self.species[i].rho / self.species[j].rho


    def update_V_col(self):
        N = len(self.species)
        for i in range(N):
            for j in range(N):
                self.V_colmat[i][j][:] = self.get_col_freq(self.species[i],self.species[j],self.mat_cs[i][j])


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

    def get_dt_col(self):
        #print("init dt = ", self.dt)
        dt = np.inf
        for s in self.species:
            c = np.sqrt(s.gamma*s.Pg/s.rho)
            #print(s.name, " col: ", np.amax(abs(s.col)))
            dts = self.dx/(max(abs(s.u)) + max(abs(self.dt*s.col)) + max(c))
            if dts < dt:
                dt = dts
        #print("Adjusted dt = ", dt*self.cfl_cut)
        return dt*self.cfl_cut

    def update_dt_conservative(self):
        dt = np.inf
        for s in self.species:
            c = np.sqrt(s.gamma*s.Pg/s.rho)
            dts = self.dx/max(abs(s.u) + c)
            if dts < dt:
                dt = dts
        self.dt = dt*self.cfl_cut

    def init_crosssections(self, dir):
        self.Nspecies = len(self.species)
        self.mat_cs = [[0] * self.Nspecies for i in range(self.Nspecies)]
        for i,sa in enumerate(self.species):
            for j,sb in enumerate(self.species):
                cs = CrossSection(dir,sa.molecule,sb.molecule)
                self.mat_cs[i][j] = cs




    def update_conservative(self):
        
        self.update_dt_conservative()
        
        
        #Collisions first for all species so that time is the same for u,rho when finding collisions
        for i,s in enumerate(self.species):
            
            s.col = np.zeros(s.lenx)
            s.Q_col = np.zeros(s.lenx)
            if self.coupled:
                self.update_V_col()
                for j,sb in enumerate(self.species):
                    if self.col:
                        s.col += self.get_col(s,sb,self.mat_cs[i][j])
                    if self.Q_col:
                        s.Q_col += self.get_Q_col(s,sb,self.mat_cs[i][j])
        
        self.dt = self.get_dt_col()
        
        for i,s in enumerate(self.species):

            [u,e,rho,Pg,s.col,s.Q_col] = pad([s.u,s.e,s.rho,s.Pg,s.col,s.Q_col],self.bnd_limits,self.bnd_type)

            gamma = s.gamma
            Pg = (gamma-1)*rho*(e-0.5*u**2)

            U0 = np.array([rho, u*rho, e*rho]).copy()
            dF = self.flux(s,U0.copy(),u.copy(),e.copy(),rho.copy(),Pg.copy())
            
            U = U0 - self.dt/self.dx * dF
            #print(f"{U0=}")
            #print(f"{dF=}")
            #print(f"{U=}")

            rho = U[0]
            u = U[1] / rho - s.col*self.dt
            e = U[2] / rho - s.Q_col*self.dt
            Pg = (s.gamma-1)*rho*(e-0.5*u**2)
            
            [s.u,s.e,s.rho,s.Pg] = unpad([u,e,rho,Pg],self.bnd_limits) 
        
        self.t += self.dt
        

    def flux_Roe(self,s,U0,u0,e0,rho0,Pg0):
        rho = U0[0]
        u = U0[1] / rho
        e = U0[2] / rho
        Pg = Pg0.copy()

        htot = s.gamma/(s.gamma-1)*Pg/rho+0.5*u**2

        Phi = np.zeros((3,self.lenx))
        rho1 = np.roll(rho,-1)
        u1 = np.roll(u,-1)
        e1 = np.roll(e,-1)
        htot1 = np.roll(htot,-1)

        #wdiff = np.roll(U0,-1,axis=1) - U0
        wdiff = np.array([rho1-rho, u1*rho1 - u*rho, e1*rho1 - e*rho])

        for i in range(0,self.lenx):
            #Roe averages
            R = np.sqrt(rho1[i]/rho[i])
            rmoy = R*rho[i]
            umoy = (R*u1[i]+u[i]) / (R+1)
            hmoy = (R*htot1[i]+htot[i] ) / (R+1)
            amoy = np.sqrt((s.gamma-1)*(hmoy-0.5*umoy**2))

            #Aux varibles
            alph1 = (s.gamma-1)*umoy**2/(2*amoy**2)
            alph2 = (s.gamma-1)/(amoy**2)

           

            #Compute vector
            #wdiff = np.roll(U0,1, axis=1)[:,i]-U0[:,i]
            #wdiff = np.array(  
            # Compute matrix P^{-1}_{j+1/2}
            Pinv = np.array([[0.5*(alph1+umoy/amoy), -0.5*(alph2*umoy+1/amoy),  alph2/2],
                            [1-alph1,                alph2*umoy,                -alph2 ],
                            [0.5*(alph1-umoy/amoy),  -0.5*(alph2*umoy-1/amoy),  alph2/2]])
                    
            # Compute matrix P_{j+1/2}
            P    = np.array([[1,              1,              1               ],
                            [umoy-amoy,        umoy,           umoy+amoy      ],
                            [hmoy-amoy*umoy,   0.5*umoy*umoy,  hmoy+amoy*umoy ]])
            
            # Compute matrix Lambda_{j+1/2}
            lamb = np.array([[ abs(umoy-amoy),  0,              0                 ],
                            [0,                 abs(umoy),      0                 ],
                            [0,                 0,              abs(umoy+amoy)    ]])
                        
            # Compute Roe matrix |A_{j+1/2}|
            A=np.dot(P,lamb)
            A=np.dot(A,Pinv)
            
            # Compute |A_{j+1/2}| (W_{j+1}-W_j)
            Phi[:,i]=np.dot(A,wdiff[:,i])
            
        # Compute Phi=(F(W_{j+1}+F(W_j))/2-|A_{j+1/2}| (W_{j+1}-W_j)/2
        F = self.func_flux(U0,s)
        #print("Phi",Phi)
        Phi=0.5*(np.roll(F,-1,axis=1)+F)-0.5*Phi
        
        dF = np.roll(Phi,-1,axis=1) - Phi
        #print("dF",dF)
        return np.roll(dF,1,axis=1) #Roll to offset axis back. There is some spooky trickery with the indeces here

    
    def func_flux(self,U,s):
        rho = U[0]
        u = U[1] / rho
        e = U[2] / rho
        Pg = (s.gamma-1)*rho*(e-0.5*u**2)
        flux_cont = np.array(rho*u)
        flux_mom = np.array(rho*u**2+Pg)
        flux_e = np.array(u*(e*rho + Pg))
        flux = np.array([flux_cont, flux_mom, flux_e])
        return flux


    def flux_Lax_Wendroff(self,s,U0,u,e,rho,Pg):

        flux_cont = np.array(rho*u)
        flux_mom = np.array(rho*u**2 + Pg)
        flux_e = np.array(u*(e*rho + Pg))
        flux = np.array([flux_cont, flux_mom, flux_e]) #Fluxes
        

        Um = np.roll(U0,1)
        Up = np.roll(U0,-1)
        fluxm = np.roll(flux,1)
        fluxp = np.roll(flux,-1)

        UpHalf = 0.5*(Up+U0) - 0.5*self.dt*(fluxp-flux) / self.dx
        UmHalf = 0.5*(Um+U0) - 0.5*self.dt*(-fluxm+flux) / self.dx


        # Corrector step
        rho= UpHalf[0]
        u = UpHalf[1] / rho
        e = UpHalf[2] / rho
        Pg = (s.gamma-1)*rho*(e-0.5*u**2)
        flux_cont = np.array(rho*u)
        flux_mom = np.array(rho*u**2 + Pg)
        flux_e = np.array(u*(e*rho + Pg))
        flux_UpHalf = np.array([flux_cont, flux_mom, flux_e])

        rho= UmHalf[0]
        u = UmHalf[1] / rho
        e = UmHalf[2] / rho
        Pg = (s.gamma-1)*rho*(e-0.5*u**2)
        flux_cont = np.array(rho*u)
        flux_mom = np.array(rho*u**2 + Pg)
        flux_e = np.array(u*(e*rho + Pg) )
        flux_UmHalf = np.array([flux_cont, flux_mom, flux_e])

        dF = flux_UpHalf - flux_UmHalf
        return dF

    def update_diffusive(self,NS=False):

        #Find common dt:
        
        self.dt = self.get_dt()
        
        
        #Collisions first for all species so that time is the same for u,rho when finding collisions
        for i,s in enumerate(self.species):
            
            s.col = np.zeros(s.lenx)
            s.Q_col = np.zeros(s.lenx)
            if self.coupled:
                self.update_V_col()
                for j,sb in enumerate(self.species):
                    if self.col:
                        s.col += self.get_col(s,sb,self.mat_cs[i][j])
                    if self.Q_col:
                        s.Q_col += self.get_Q_col(s,sb,self.V_colmat[i,j,:])
        
        self.dt = self.get_dt_col()
        
        for s in self.species:

            [u,e,rho,Pg,s.col,s.Q_col] = pad([s.u,s.e,s.rho,s.Pg,s.col,s.Q_col],self.bnd_limits,self.bnd_type)

            gamma = s.gamma

            c = np.sqrt(gamma*Pg/rho) #Can't be zero because pressure/density can't be zero

            if NS:
                mu_dyn = self.get_mu_dyn(s)
                [mu_dyn] = pad([mu_dyn],self.bnd_limits,self.bnd_type)
                u = step_momentum_NS(self.xx,u,rho,Pg,self.dt,mu_dyn,cq = s.cq, cL = s.cL, col=s.col,ddx=self.ddx) / rho
            else:
                u = step_momentum(self.xx,u,rho,Pg,self.dt,cq = s.cq, cL = s.cL, col=s.col,ddx=self.ddx) / rho
 
            if True:
                for j in range(len(u)):
                    if np.abs(u[j]) > np.abs(c[j]) and u[j] > 0:
                        u[j] = c[j]
                    elif np.abs(u[j]) > np.abs(c[j]) and u[j] < 0:
                        u[j] = -c[j]

            e = step_energy(self.xx,e,u,Pg,rho,self.dt,cq = s.cq, cL = s.cL, Q_col = s.Q_col,ddx=self.ddx)
            rho = step_density(self.xx,u,rho,self.dt,cD = s.cD, ddx=self.ddx)
            Pg = step_pressure(Pg,e,rho,u,gamma)


            [s.u,s.e,s.rho,s.Pg] = unpad([u,e,rho,Pg],self.bnd_limits) 
        self.t += self.dt

    def get_mu_dyn(self,s):

        l = self.k_b*s.T / (np.sqrt(2)*s.sigma*s.Pg)
        print("T = ",np.amax(s.T))
        alpha = 1
        mu = alpha * s.rho * l * np.sqrt(s.T*2*self.k_b/(np.pi*s.m))
        print("mu = ",np.amax(np.abs(mu)))
        return mu
    
    def get_col(self, sa, sb,cs):
        col_freq = self.get_col_freq(sa,sb,cs)
        return sa.rho*col_freq*(sa.u-sb.u)
        #return sa.rho * self.get_col_freq(sa,sb)*(sa.u-sb.u)


    def get_col_freq(self, sa,sb,cs):

        T_ab = (sa.m*sa.T + sb.m*sb.T) / (sa.m+sb.m)
        m_ab = (sa.m*sb.m)/(sa.m+sb.m)
        
        mu_a = np.zeros(len(sa.T))
        mu_b = np.zeros(len(sa.T))
        mu_ab = np.zeros(len(sa.T))
        uth_ab = np.zeros(len(sa.T))
        for i in range(len(sa.T)):
            if sa.T[i] > 1e-20 and sb.T[i] > 1e-20:
                mu_a[i] = sa.m/(self.k_b*sa.T[i])
                mu_b[i] = sb.m/(self.k_b*sb.T[i])
                mu_ab[i] = mu_a[i]*mu_b[i] / (mu_a[i]+mu_b[i])
                uth_ab[i] = np.sqrt(8/(np.pi*np.abs(mu_ab[i])))
            
        n_b = sb.rho/sb.m
        sigma = cs.get_sigma(T_ab)
        #print("sigma",np.amax(sigma))
        col_freq = (m_ab/sa.m) * n_b * uth_ab * sigma
        return col_freq


    def get_Q_col(self, sa, sb, cs):
        v_ab = self.get_col_freq(sa,sb,cs)
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
