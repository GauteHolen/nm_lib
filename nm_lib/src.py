#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 02 10:25:17 2021

@author: Juan Martinez Sykora

"""

# import builtin modules
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# import external public "common" modules
import numpy as np
import matplotlib.pyplot as plt 

def animMult(uts,xx,lbls, styles, t,n_frames=100, nt = False, log_time = False,ylim = None):
    """
    animate ut(xx) in time with a limited number of timesteps

    Option for log spaced becuase more than about 100 timesteps animated becomes very slow...
    
    Parameters
    ----------
    ut : `array [array]` 
        Array of each array of points u(x) in time
    xx : `array` 
        spacial axis


    n_frames : `int` 
        Number of frames in the animation
    log_time : `boolean`
        if true the timestep scales logarithmic and not linearly

    Returns
    ------- 
    `HTML`
        HTML animation object
    """
    def init():
        for ut, lbl, style in zip(uts,lbls,styles):
            axes.plot(xx,ut[0], label = lbl, linestyle = style)
            if ylim:
                axes.set_ylim(ylim)
        plt.legend()

    def animate(i):
        axes.clear()
        for ut, lbl, style in zip(uts,lbls,styles):
            alpha = 1
            
            if ylim:
                axes.set_ylim(ylim)
                umax = np.abs(np.amax(ut[i]))
                if umax > np.amax(ylim):
                    alpha = min(1/np.sqrt(umax)+0.25,1)
            
                axes.plot(xx,ut[i], label = lbl, linestyle = style,alpha=alpha)
            else:
                axes.plot(xx,ut[i], label = lbl, linestyle = style,alpha=alpha)
        if nt:
            axes.set_title(f'Timestep={i}')
        else:
            axes.set_title('t=%.2f'%t[i])
        plt.legend()

    Nt = uts[0].shape[0]

    if log_time:
        frames = np.zeros(n_frames, dtype=np.int64)
        #First some linearly spaced frames
        lin_frames = int(0.20*n_frames)
        frames[0:lin_frames] = np.linspace(0,lin_frames-1,lin_frames, dtype=np.int64)
        #Log spaced frames
        frames[lin_frames:] = np.geomspace(lin_frames,Nt-1,num=n_frames-lin_frames,dtype=np.int64)[:]
    else:
        frames = np.linspace(1,Nt-1,num=n_frames, dtype=np.int64)

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
    anim = FuncAnimation(fig, animate, interval=20, frames=frames, init_func=init)
    html = HTML(anim.to_jshtml())
    plt.close()
    return html

def animMultX(uts,xxs,lbls, styles, t,n_frames=100,ylim = None, nt = False, log_time = False):
    """
    animate ut(xx) in time with a limited number of timesteps

    Option for log spaced becuase more than about 100 timesteps animated becomes very slow...
    
    Parameters
    ----------
    ut : `array [array]` 
        Array of each array of points u(x) in time
    xx : `array` 
        spacial axis


    n_frames : `int` 
        Number of frames in the animation
    log_time : `boolean`
        if true the timestep scales logarithmic and not linearly

    Returns
    ------- 
    `HTML`
        HTML animation object
    """
    def init():
        for ut, xx, lbl, style in zip(uts,xxs,lbls,styles):
            axes.plot(xx,ut[0], label = lbl, linestyle = style)
            if ylim:
                axes.set_ylim(ylim)
        plt.legend()

    def animate(i):
        axes.clear()
        for ut,xx, lbl, style in zip(uts,xxs,lbls,styles):
            axes.plot(xx,ut[i], label = lbl, linestyle = style)
            if ylim:
                axes.set_ylim(ylim)
        if nt:
            axes.set_title(f'Timestep={i}')
        else:
            axes.set_title('t=%.2f'%t[i])
        plt.legend()

    Nt = uts[0].shape[0]

    if log_time:
        frames = np.zeros(n_frames, dtype=np.int64)
        #First some linearly spaced frames
        lin_frames = int(0.20*n_frames)
        frames[0:lin_frames] = np.linspace(0,lin_frames-1,lin_frames, dtype=np.int64)
        #Log spaced frames
        frames[lin_frames:] = np.geomspace(lin_frames,Nt-1,num=n_frames-lin_frames,dtype=np.int64)[:]
    else:
        frames = np.linspace(1,Nt-1,num=n_frames, dtype=np.int64)

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
    anim = FuncAnimation(fig, animate, interval=20, frames=frames, init_func=init)
    html = HTML(anim.to_jshtml())
    plt.close()
    return html


def anim(ut,xx,t,n_frames=100, log_time = False):
    """
    animate ut(xx) in time with a limited number of timesteps

    Option for log spaced becuase more than about 100 timesteps animated becomes very slow...
    
    Parameters
    ----------
    ut : `array [array]` 
        Array of each array of points u(x) in time
    xx : `array` 
        spacial axis


    n_frames : `int` 
        Number of frames in the animation
    log_time : `boolean`
        if true the timestep scales logarithmic and not linearly

    Returns
    ------- 
    `HTML`
        HTML animation object
    """
    def init(): 
        axes.plot(xx,ut[0])

    def animate(i):
        axes.clear()
        axes.plot(xx,ut[i])
        axes.set_title('t=%.2f'%t[i])

    Nt = len(t)

    if log_time:
        frames = np.zeros(n_frames, dtype=np.int64)
        #First some linearly spaced frames
        lin_frames = int(0.20*n_frames)
        frames[0:lin_frames] = np.linspace(0,lin_frames-1,lin_frames, dtype=np.int64)
        #Log spaced frames
        frames[lin_frames:] = np.geomspace(lin_frames,Nt-1,num=n_frames-lin_frames,dtype=np.int64)[:]
    else:
        frames = np.linspace(0,Nt-1,num=n_frames, dtype=np.int64)

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
    anim = FuncAnimation(fig, animate, interval=20, frames=frames, init_func=init)
    html = HTML(anim.to_jshtml())
    plt.close()
    return html

def deriv_dnw(xx, hh, dtype = np.float64, **kwargs):
    """
    Returns the downwind 2nd order derivative of hh array respect to xx array. 

    Parameters 
    ----------
    xx : `array`
        Spatial axis. 
    hh : `array`
        Function that depends on xx. 

    Returns
    -------
    `array`
        The downwind 2nd order derivative of hh respect to xx. Last 
        grid point is ill (or missing) calculated. 
    """
    dx = np.empty(xx.shape, dtype = dtype)
    dh = np.empty(hh.shape, dtype = dtype)
    dh[:-1] = hh[1:]-hh[0:-1]
    dx[:-1] = xx[1:]-xx[0:-1]
    dh[-1] = dh[-2]
    dx[-1] = dx[-2]
    return dh/dx



def order_conv(hh, hh2, hh4, **kwargs):
    """
    Computes the order of convergence of a derivative function 

    Parameters 
    ----------
    hh : `array`
        Function that depends on xx. 
    hh2 : `array`
        Function that depends on xx but with twice number of grid points than hh. 
    hh4 : `array`
        Function that depends on xx but with twice number of grid points than hh2.
    Returns
    -------
    `array` 
        The order of convergence.  
    """

    
    frac = np.ones(hh.shape) #init
    for i in range(len(hh)):
        frac[i] = (hh4[i*4]-hh2[i*2])/(hh2[i*2]-hh[i])

    #Absolute value to prevent runtime error of log of negative numbers
    m = np.ma.log(frac)/np.log(2)

    return m

def deriv_4tho(xx, hh,**kwargs): 
    """
    Returns the 4th order derivative of hh respect to xx.

    Parameters 
    ---------- 
    xx : `array`
        Spatial axis. 
    hh : `array`
        Function that depends on xx. 

    Returns
    -------
    `array`
        The centered 4th order derivative of hh respect to xx. 
        Last and first two grid points are ill calculated. 
    """
    dx = xx[1]-xx[0]
    #setting up the array
    dhdx = np.zeros(hh.shape, dtype=type(hh[0]))

    #First 2 elements
    dhdx[0]= 8*hh[1] - hh[2] 
    dhdx[1]= -8*hh[0] + 8*hh[2] - hh[3]

    #Bulk of elements
    dhdx[2:-2] = hh[0:-4] - 8*hh[1:-3] + 8*hh[3:-1] - hh[4:]

    #Last 2 elements
    dhdx[-2] = hh[-4] - 8*hh[-3] + 8*hh[-1]
    dhdx[-1] = hh[-3] - 8*hh[-2]
    #Divide by 12*dx
    dhdx = dhdx /(12*dx)

    return dhdx


   

def step_adv_burgers(xx, hh, a, cfl_cut = 0.98, 
                    ddx = lambda x,y: deriv_dnw(x, y), **kwargs): 
    r"""
    Right hand side of Burger's eq. where a can be a constant or a function that 
    depends on xx. 

    Requires 
    ---------- 
    cfl_adv_burger function which computes np.min(dx/a)

    Parameters
    ----------
    xx : `array`
        Spatial axis. 
    hh : `array`
        Function that depends on xx.
    a : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    cfl_cut : `float`
        Constant value to limit dt from cfl_adv_burger. 
        By default clf_cut=0.98. 
    ddx : `lambda function`
        Allows to select the type of spatial derivative. 
        By default lambda x,y: deriv_dnw(x, y)

    Returns
    -------
    `array` 
        Time interval.
        Right hand side of (u^{n+1}-u^{n})/dt = from burgers eq, i.e., x \frac{\partial u}{\partial x} 
    """    
    u_next = np.zeros(xx.shape,dtype=type(xx[0]))

    dt = cfl_cut * cfl_adv_burger(a,xx)
    dudx = ddx(xx,hh)
    u_next =hh -a*dudx*dt
 
    return dt, u_next


def cfl_adv_burger(a,x): 
    """
    Computes the dt_fact, i.e., Courant, Fredrich, and 
    Lewy condition for the advective term in the Burger's eq. 

    Parameters
    ----------
    a : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    x : `array`
        Spatial axis. 

    Returns
    ------- 
    `float`
        min(dx/|a|)
    """
    #Array of dx
    dx = x[1:]-x[0:-1]
    if isinstance(a, np.ndarray):
        maxa = np.amax(np.abs(a))
        return np.amin(dx/maxa)
    else:
        return np.amin(dx/(np.abs(a)))


def evolv_adv_burgers(xx, hh, nt, a, cfl_cut = 0.98, 
        ddx = lambda x,y: deriv_dnw(x, y), 
        bnd_type='wrap', bnd_limits=[0,1], keep_centered = False,**kwargs):
    r"""
    Advance nt time-steps in time the burger eq for a being a a fix constant or array.
    Requires
    ----------
    step_adv_burgers

    Parameters
    ----------
    xx : `array`
        Spatial axis. 
    hh : `array`
        Function that depends on xx.
    a : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    cfl_cut : `float`
        Constant value to limit dt from cfl_adv_burger. 
    ddx : `lambda function`
        Allows to change the space derivative function.
        By default lambda x,y: deriv_dnw(x, y).  
    bnd_type : `string`
        Allows to select the type of boundaries. 
        By default 'wrap'.
    bnd_limits : `list(int)`
        Array of two integer elements. The number of pixels that
        will need to be updated with the boundary information. 
        By default [0,1].
    keep_centered : `int` or `False`
        If int the program will keep the curve from drifting by
        applying padding with boundary conditions before and 
        after the differentiation and do interpolation to correct
        for drift prop to dx*dt

    Returns
    ------- 
    t : `array`
        time 1D array
    unnt : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain. 
    """
    dt = cfl_cut * cfl_adv_burger(a,xx)
    dx = np.mean([x2-x1 for x2,x1 in zip(xx[1:],xx[0:-1])])
    tt = np.linspace(0,(nt+1)*dt, nt)
    x_len = xx.shape[0]

    if keep_centered is not False and type(keep_centered) is int:
        xx_cent = xx[:]+dx*dt*keep_centered*-1

    uunt = np.zeros((nt,x_len))
    uunt[0] = hh
    u = hh
    #u = np.pad(u,bnd_limits,bnd_type)
    for i in range(1,len(tt)):
        if keep_centered is not False and type(keep_centered) is int:
            #Drift due to dnw method
            u_pad = np.pad(u,[x+2 for x in bnd_limits],bnd_type)
            x_pad = np.pad(xx,[x+2 for x in bnd_limits],'reflect', reflect_type='odd')
            _,u_step = step_adv_burgers(x_pad,u_pad,a,ddx=ddx,cfl_cut=cfl_cut)
            
            if ddx == deriv_upw:
                u_next = u_step[3+bnd_limits[0]:-1-bnd_limits[1]]
                
            elif ddx == deriv_dnw:
                #u_next = u_step[1+bnd_limits[0]:-3-bnd_limits[1]]
                u_next = u_step[1:-4]
            else:
                u_next = u_step[2:-3] #centered
                

            #Drift in space proportional to dx*dt per timestep
            u_next = np.interp(xx_cent,xx,u_next)

        else:
            _,u_next = step_adv_burgers(xx,u,a,ddx=ddx,cfl_cut=cfl_cut)
            if bnd_limits[1] == 0:
                u_next = np.pad(u_next[bnd_limits[0]:],bnd_limits,bnd_type)
            else:
                u_next = np.pad(u_next[bnd_limits[0]:-bnd_limits[1]],bnd_limits,bnd_type)
        #u_next[0] = u_next[-1]       
        uunt[i] = u_next
        u = u_next
    #print(len(tt), dt, x_len, dx)
    return tt, uunt



def deriv_upw(xx, hh, dtype = np.float64, **kwargs):
    r"""
    returns the upwind 2nd order derivative of hh respect to xx. 

    Parameters
    ----------
    xx : `array`
        Spatial axis. 
    hh : `array`
        Function that depends on xx. 

    Returns
    ------- 
    `array`
        The upwind 2nd order derivative of hh respect to xx. First 
        grid point is ill calculated. 
    """

    dx = np.empty(xx.shape, dtype = dtype)
    dh = np.empty(xx.shape, dtype = dtype)
    dh[1:] = hh[1:]-hh[0:-1]
    dx[1:] = xx[1:]-xx[0:-1]
    dh[0] = dh[1]
    dx[0] = dx[1]
    return dh/dx

    

def deriv_cent(xx, hh, **kwargs):
    r"""
    returns the centered 2nd derivative of hh respect to xx. 

    Parameters
    ---------- 
    xx : `array`
        Spatial axis. 
    hh : `array`
        Function that depends on xx. 

    Returns
    -------
    `array`
        The centered 2nd order derivative of hh respect to xx. First 
        and last grid points are ill calculated. 
    """

    dhdx = np.zeros(hh.shape, dtype=type(xx[0]))

    #First and last terms
    dhdx[0] = (hh[1]-hh[0])/(xx[1]-xx[0])
    dhdx[-1] = (hh[-1]-hh[-2])/(xx[-1]-xx[-2])

    dhdx[1:-1] = (hh[0:-2]-hh[2:]) / (xx[0:-2]-xx[2:])
    
    return dhdx


def deriv_d2dx2(xx,hh):
    r"""
    returns the second centered 2nd derivative of hh respect to xx. 

    Parameters
    ---------- 
    xx : `array`
        Spatial axis. 
    hh : `array`
        Function that depends on xx. 

    Returns
    -------
    `array`
        The centered 2nd order derivative of hh respect to xx. First 
        and last grid points are ill calculated. 
    """

    dx = xx[1] - xx[0]
    d2dx2 = np.roll(hh,1) - 2*hh + np.roll(hh,-1)

    return d2dx2/(dx*dx)



def evolv_uadv_burgers(xx, hh, nt, cfl_cut = 0.98, 
        ddx = lambda x,y: deriv_dnw(x, y), 
        bnd_type='wrap', diff = False, bnd_limits=[0,1], tf = None,**kwargs):
    r"""
    Advance nt time-steps in time the burger eq for a being u.

    Requires
    --------
    step_uadv_burgers

    Parameters
    ----------
    xx : `array`
        Spatial axis. 
    hh : `array`
        Function that depends on xx.
    cfl_cut : `float`
        constant value to limit dt from cfl_adv_burger. 
        By default 0.98.
    ddx : `lambda function` 
        Allows to change the space derivative function. 
    bnd_type : `string` 
        It allows to select the type of boundaries.
        By default 'wrap'
    bnd_limits : `list(int)`
        List of two integer elements. The number of pixels that
        will need to be updated with the boundary information.
        By default [0,1]

    Returns
    -------
    t : `array` 
        Time 1D array
    unnt : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain. 
    """

    
    #dx = np.mean([x2-x1 for x2,x1 in zip(xx[1:],xx[0:-1])])
    x_len = xx.shape[0]
    tt = np.array([0])

    uunt = np.zeros((nt,x_len))
    uunt[0] = hh
    u = hh
    if tf:
        nt = int(1e30)
    t = 0
    #u = np.pad(u,bnd_limits,bnd_type)
    for i in range(1,nt):
        
        if diff:
            dt = cfl_diff_burger(u,xx) * cfl_cut
            step = step_diff_burgers(xx,u,u)
            u_next = u + dt*step
        else:
            dt = cfl_cut * cfl_adv_burger(u,xx)
            step = step_uadv_burgers(xx,u,ddx=ddx,cfl_cut=cfl_cut)
            u_next = u - u*dt*step
        if bnd_limits[1] == 0:
            u_next = np.pad(u_next[bnd_limits[0]:],bnd_limits,bnd_type)
        else:
            u_next = np.pad(u_next[bnd_limits[0]:-bnd_limits[1]],bnd_limits,bnd_type)
        #u_next[0] = u_next[-1]       
        uunt[i] = u_next
        u = u_next
        t+=dt
        tt = np.append(tt,t)
        if tf:
            if t > tf:
                break
    #print(len(tt), dt, x_len, dx)
    return tt, uunt

def step_Rie_adv_burgers(xx, hh, cfl_cut = 0.98, 
                    ddx = lambda x,y: deriv_dnw(x, y), **kwargs): 
    r"""
    Right hand side of Burger's eq. where a can be a constant or a function that 
    depends on xx. 

    Requires 
    ---------- 
    cfl_adv_burger function which computes np.min(dx/a)

    Parameters
    ----------
    xx : `array`
        Spatial axis. 
    hh : `array`
        Function that depends on xx.
    a : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    cfl_cut : `float`
        Constant value to limit dt from cfl_adv_burger. 
        By default clf_cut=0.98. 
    ddx : `lambda function`
        Allows to select the type of spatial derivative. 
        By default lambda x,y: deriv_dnw(x, y)

    Returns
    -------
    `array` 
        Time interval.
        Right hand side of (u^{n+1}-u^{n})/dt = from burgers eq, i.e., x \frac{\partial u}{\partial x} 
    """    
    
    uL = hh
    uR = np.roll(hh,1)
    v_a = np.array([max(u,um1) for u,um1 in zip(hh,np.roll(hh,-1))])
    #v_a = get_v_a(hh)
    flux = flux_rusanov(uL,uR, v_a)

    
    dt = cfl_cut * cfl_adv_burger(v_a, xx)
    u_next = hh - dt*(flux - np.roll(flux,1))
    u_next = np.pad(u_next[1:-1], [1, 1], "wrap")
 
    return dt, u_next

def flux_rusanov(uL,uR, v_a):
    flux_L = uL * uL * 0.5
    flux_R = uR * uR * 0.5
    flux = (flux_L + flux_R) * 0.5 - 0.5 * v_a * (uR - uL)
    return flux

def get_v_a(hh):
    #arr1 = np.abs(hh)
    #arr2 = np.abs(np.roll(hh,-1)) 
    arr1 = hh
    arr2 = np.roll(hh,-1)
    s= np.max([arr1[:],arr2[:]])
    print(s)
    return s


def evolv_Rie_uadv_burgers(xx, hh, nt, tf=None, cfl_cut = 0.98, 
        ddx = lambda x,y: deriv_dnw(x, y), 
        bnd_type='wrap', bnd_limits=[1,1], **kwargs):
    r"""
    Advance nt time-steps in time the burger eq for a being u using the Lax method.

    Requires
    -------- 
    step_uadv_burgers

    Parameters
    ----------
    xx : `array`
        Spatial axis. 
    hh : `array`
        Function that depends on xx.
    cfl_cut : `array`
        Constant value to limit dt from cfl_adv_burger. 
        By default 0.98
    ddx : `array`
        Lambda function allows to change the space derivative function.
        By derault  lambda x,y: deriv_dnw(x, y)
    bnd_type : `string`
        It allows to select the type of boundaries 
    bnd_limits : `list(int)`
        List of two integer elements. The number of pixels that
        will need to be updated with the boundary information. 
        By default [0,1]

    Returns
    -------
    t : `array`
        Time 1D array
    unnt : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain. 
    """
    print("Riemann solver")
    
    x_len = xx.shape[0]

    uunt = np.zeros((nt,x_len))
    tt = np.zeros(nt)

    uunt[0] = hh


    for i in range(nt-1):
        dt, step = step_Rie_adv_burgers(xx,uunt[i])
        uunt[i+1] = step
        tt[i+1] = tt[i] + dt
        if tf and tt[i]>tf:
            print(f"exceeded final time {tf:.3f} at {tt[i+1]:.3f}")
            uunt = uunt[0:i+1]
            tt = tt[0:i+1]
            break

    return tt,uunt


def evolv_Lax_uadv_burgers(xx, hh, nt, cfl_cut = 0.98, 
        ddx = lambda x,y: deriv_dnw(x, y), 
        bnd_type='wrap', bnd_limits=[1,1], tf=None, **kwargs):
    r"""
    Advance nt time-steps in time the burger eq for a being u using the Lax method.

    Requires
    -------- 
    step_uadv_burgers

    Parameters
    ----------
    xx : `array`
        Spatial axis. 
    hh : `array`
        Function that depends on xx.
    cfl_cut : `array`
        Constant value to limit dt from cfl_adv_burger. 
        By default 0.98
    ddx : `array`
        Lambda function allows to change the space derivative function.
        By derault  lambda x,y: deriv_dnw(x, y)
    bnd_type : `string`
        It allows to select the type of boundaries 
    bnd_limits : `list(int)`
        List of two integer elements. The number of pixels that
        will need to be updated with the boundary information. 
        By default [0,1]

    Returns
    -------
    t : `array`
        Time 1D array
    unnt : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain. 
    """
    print("lax method")


    dx = xx[1]-xx[0] #Maybe not the best

    tt = np.array([0])

    uunt = np.zeros((nt,xx.shape[0]))
    uunt[0] = hh

    if tf:
        nt = int(1e30)
    t = 0
    for i in range(1,nt):
        dt = cfl_cut * dx / np.amax(hh)
        h_avg = np.zeros(hh.shape, dtype=type(hh[0]))
        h_avg[1:-1] = hh[0:-2]+hh[2:]
        h_avg[0] = hh[0]+hh[1]
        h_avg[-1] = hh[-1]+hh[-2]
        h_avg *= 0.5

        h_avg2 = np.zeros(hh.shape, dtype=type(hh[0]))
        h_avg2[1:-1] = -hh[0:-2]+hh[2:]
        h_avg2[0] = -hh[0]+hh[1]
        h_avg2[-1] = -hh[-1]+hh[-2]
        h_avg2 *= 0.5

        hh_new = h_avg -dt/dx * hh*h_avg2
        hh_new = np.pad(hh_new[bnd_limits[0]:-bnd_limits[1]],bnd_limits,bnd_type)
        uunt[i] = hh_new      
        hh = hh_new
        t+=dt
        tt = np.append(tt,t)
        if tf:
            if t > tf:
                break

    return tt, uunt




def step_Lax_uadv_burgers(hh):
    """Computes the averages of u along x for the lax method
       hh_avg1 = 0.5 ( u^n_{j+1} + u^n_{j-1} )
       hh_avg2 = 0.5 ( u^n_{j+1} - u^n_{j-1} )

    Args:
        hh (array): Depends on x

    Returns:
        _type_: _description_
    """

    h_avg1 = np.roll(hh,1) + np.roll(hh,-1)
    h_avg2 = np.roll(hh,1) - np.roll(hh,-1)

    return 0.5*h_avg1, 0.5*h_avg2

def evolv_Lax_adv_burgers(xx, hh, nt, a = 1, cfl_cut = 0.98, 
        ddx = lambda x,y: deriv_dnw(x, y), 
        bnd_type='wrap', bnd_limits=[1,1], tf=None, **kwargs):
    r"""
    Advance nt time-steps in time the burger eq for a being a a fix constant or array.

    Requires
    --------
    step_adv_burgers

    Parameters
    ----------
    xx : `array`
        Spatial axis. 
    hh : `array`
        Function that depends on xx.
    nt : `int`
        Number of iterations
    a : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    cfl_cut : `float`
        Constant value to limit dt from cfl_adv_burger. 
        By default 0.98
    ddx : `lambda function` 
        Allows to change the space derivative function. 
        By default lambda x,y: deriv_dnw(x, y)
    bnd_type : `string` 
        It allows to select the type of boundaries. 
        By default 'wrap'
    bnd_limits : `list(int)`
        Array of two integer elements. The number of pixels that
        will need to be updated with the boundary information. 
        By default [0,1]

    Returns
    -------
    t : `array`
        Time 1D array
    unnt : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain. 
    """

    dx = xx[1]-xx[0] #Maybe not the best

    tt = np.array([0])

    uunt = np.zeros((nt,xx.shape[0]))
    uunt[0] = hh

    if tf:
        nt = int(1e30)
    t = 0
    for i in range(1,nt):
        dt = cfl_cut * dx / np.amax(hh)
        h_avg = np.zeros(hh.shape, dtype=type(hh[0]))
        h_avg[1:-1] = hh[0:-2]+hh[2:]
        h_avg[0] = hh[0]+hh[1]
        h_avg[-1] = hh[-1]+hh[-2]
        h_avg *= 0.5

        h_avg2 = np.zeros(hh.shape, dtype=type(hh[0]))
        h_avg2[1:-1] = -hh[0:-2]+hh[2:]
        h_avg2[0] = -hh[0]+hh[1]
        h_avg2[-1] = -hh[-1]+hh[-2]
        h_avg2 *= 0.5

        hh_new = h_avg -dt/dx * a*h_avg2
        hh_new = np.pad(hh_new[bnd_limits[0]:-bnd_limits[1]],bnd_limits,bnd_type)
        uunt[i] = hh_new      
        hh = hh_new
        t+=dt
        tt = np.append(tt,t)
        if tf:
            if t > tf:
                break

    return tt, uunt


def step_uadv_burgers(xx, hh, cfl_cut = 0.98, 
                    ddx = lambda x,y: deriv_dnw(x, y), **kwargs): 
    r"""
    Right hand side of Burger's eq. where a is u, i.e hh.  

    Requires
    --------
        cfl_adv_burger function which computes np.min(dx/a)

    Parameters
    ----------   
    xx : `array`
        Spatial axis. 
    hh : `array`
        Function that depends on xx.
    cfl_cut : `array`
        Constant value to limit dt from cfl_adv_burger. 
        By default 0.98
    ddx : `lambda function` 
        Allows to select the type of spatial derivative.
        By default lambda x,y: deriv_dnw(x, y)


    Returns
    -------
    dt : `array`
        time interval
    unnt : `array`
        right hand side of (u^{n+1}-u^{n})/dt = from burgers eq, i.e., x \frac{\partial u}{\partial x} 
    """

    u_next = np.zeros(xx.shape,dtype=type(xx[0]))
    dudx = ddx(xx,hh)
    u_next =dudx
 
    return u_next       


def cfl_diff_burger(a,xx): 
    r"""
    Computes the dt_fact, i.e., Courant, Fredrich, and 
    Lewy condition for the diffusive term in the Burger's eq. 

    Parameters
    ----------
    a : `float` or `array` 
        Either constant, or array which multiply the right hand side of the Burger's eq.
    x : `array`
        Spatial axis. 

    Returns
    -------
    `float`
        min(dx/|a|)
    """
    dx = xx[1] - xx[0]

    return 0.5*dx*dx/np.amax(a)


def ops_Lax_LL_Add(xx, hh, nt, a, b, cfl_cut = 0.98, 
        ddx = lambda x,y: deriv_cent(x, y), 
        bnd_type='wrap', bnd_limits=[0,1], **kwargs): 
    r"""
    Advance nt time-steps in time the burger eq for a being a and b 
    a fix constant or array. Solving two advective terms separately 
    with the Additive Operator Splitting scheme.  Both steps are 
    with a Lax method. 

    Requires
    --------
    step_adv_burgers
    cfl_adv_burger

    Parameters
    ----------
    xx : `array`
        Spatial axis. 
    hh : `array`
        Function that depends on xx.
    nt : `int`
        Number of iterations
    a : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    b : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    cfl_cut : `float`
        Constant value to limit dt from cfl_adv_burger. 
        By default 0.98
    ddx : `lambda function` 
        Allows to change the space derivative function. 
        By default lambda x,y: deriv_dnw(x, y)
    bnd_type : `string` 
        It allows to select the type of boundaries 
        By default 'wrap'
    bnd_limits : `list(int)`
        List of two integer elements. The number of pixels that
        will need to be updated with the boundary information. 
        By default [0,1]

    Returns
    -------
    t : `array`
        Time 1D array
    unnt : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain. 
    """

    print("add OS method")


    tt = np.zeros(nt)
    uunt = np.zeros((nt,xx.shape[0]))
    uunt[0] = hh
    dx = xx[1] - xx[0]

    for i in range(1,nt):
        
        #dt1, rhs1 = step_adv_burgers(xx,hh,a,cfl_cut=cfl_cut, ddx=ddx)
        #dt2, rhs2 = step_adv_burgers(xx,hh,b,cfl_cut=cfl_cut, ddx=ddx)
        
        dt1 = cfl_cut * cfl_adv_burger(a,xx)
        dt2 = cfl_cut * cfl_adv_burger(b,xx)

        hh_avg1, hh_avg2 = step_Lax_uadv_burgers(hh)

        dt = np.min([dt1,dt2])

        hh1 = hh_avg1 - a*dt/dx * hh_avg2
        hh2 = hh_avg1 - b*dt/dx * hh_avg2

        hh_new = hh1 + hh2 - hh

        hh_new = np.pad(hh_new[bnd_limits[0]:-bnd_limits[1]],bnd_limits,bnd_type)
        uunt[i] = hh_new      
        hh = hh_new
        tt[i] = tt[i-1] + dt

    return tt, uunt




def ops_Lax_LL_Lie(xx, hh, nt, a, b, cfl_cut = 0.98, 
        ddx = lambda x,y: deriv_dnw(x, y), 
        bnd_type='wrap', bnd_limits=[0,1], **kwargs): 
    r"""
    Advance nt time-steps in time the burger eq for a being a and b 
    a fix constant or array. Solving two advective terms separately 
    with the Lie-Trotter Operator Splitting scheme.  Both steps are 
    with a Lax method. 

    Requires: 
    step_adv_burgers
    cfl_adv_burger

    Parameters
    ----------
    xx : `array`
        Spatial axis. 
    hh : `array`
        Function that depends on xx.
    nt : `int`
        Number of iterations
    a : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    b : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    cfl_cut : `float` 
        Limit dt from cfl_adv_burger.
        By default 0.98
    ddx : `lambda function` 
        Allows to change the space derivative function. 
        By default lambda x,y: deriv_dnw(x, y)
    bnd_type : `string`
        It allows to select the type of boundaries. 
        By default 'wrap'
    bnd_limits : `list(int)`
        List of two integer elements. The number of pixels that
        will need to be updated with the boundary information. 
        By default [0,1]

    Returns
    -------
    t : `array`
        Time 1D array
    unnt : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain. 
    """

    print("Lie OS method")


    tt = np.zeros(nt)
    uunt = np.zeros((nt,xx.shape[0]))
    uunt[0] = hh
    dx = xx[1] - xx[0]

    for i in range(1,nt):

        
        
        dt1 = cfl_cut * cfl_adv_burger(a,xx)
        dt2 = cfl_cut * cfl_adv_burger(b,xx)
        dt = np.min([dt1,dt2])
        

        hh_avg1, hh_avg2 = step_Lax_uadv_burgers(hh)
        hh1 = hh_avg1 - a*dt/dx * hh_avg2

        hh_avg1, hh_avg2 = step_Lax_uadv_burgers(hh1)
        hh2 = hh_avg1 - b*dt2/dx * hh_avg2

        hh_new = hh2

        hh_new = np.pad(hh_new[bnd_limits[0]:-bnd_limits[1]],bnd_limits,bnd_type)
        uunt[i] = hh_new      
        hh = hh_new
        tt[i] = tt[i-1] + dt


    return tt, uunt


def ops_Lax_LL_Strang(xx, hh, nt, a, b, cfl_cut = 0.98, 
        ddx = lambda x,y: deriv_dnw(x, y), 
        bnd_type='wrap', bnd_limits=[0,1], **kwargs): 
    r"""
    Advance nt time-steps in time the burger eq for a being a and b 
    a fix constant or array. Solving two advective terms separately 
    with the Lie-Trotter Operator Splitting scheme. Both steps are 
    with a Lax method. 

    Requires
    --------
    step_adv_burgers
    cfl_adv_burger
    numpy.pad for boundaries. 

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    hh : `array`
        Function that depends on xx.
    nt : `int`
        Number of iterations
    a : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    b : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    cfl_cut : `float`
        Constant value to limit dt from cfl_adv_burger.
        By default 0.98
    ddx : `lambda function` 
        Allows to change the space derivative function.
        By default lambda x,y: deriv_dnw(x, y)
    bnd_type : `string` 
        Allows to select the type of boundaries.
        By default `wrap`
    bnd_limits : `list(int)` 
        The number of pixels that will need to be updated with the boundary information.
        By default [0,1]

    Returns
    -------
    t : `array`
        Time 1D array
    unnt : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain. 
    """

    print("Strang OS method")


    tt = np.zeros(nt)
    uunt = np.zeros((nt,xx.shape[0]))
    uunt[0] = hh
    dx = xx[1] - xx[0]

    

    for i in range(1,nt):
        
       #same subscripts as wikipedia Strang Splitting

        dt1 = cfl_cut * cfl_adv_burger(a,xx)
        dt2 = cfl_cut * cfl_adv_burger(b,xx)
        dt = np.min([dt1,dt2])

        hh_avg1, hh_avg2 = step_Lax_uadv_burgers(hh)
        hh1_tilde = hh_avg1 - 0.5 * a*dt/dx * hh_avg2     # half step
       
        hh_avg1, hh_avg2 = step_Lax_uadv_burgers(hh1_tilde) 
        hh1_bar = hh_avg1 - b*dt/dx * hh_avg2             # full step

        hh_avg1, hh_avg2 = step_Lax_uadv_burgers(hh1_bar) 
        hh1 = hh_avg1 - 0.5 * a*dt/dx * hh_avg2           # half step

        hh_avg1, hh_avg2 = step_Lax_uadv_burgers(hh1)
        hh2_tilde = hh_avg1 - 0.5 * a*dt/dx * hh_avg2     # half step
       
        hh_avg1, hh_avg2 = step_Lax_uadv_burgers(hh2_tilde) 
        hh2_bar = hh_avg1 - b*dt/dx * hh_avg2             # full step

        hh_avg1, hh_avg2 = step_Lax_uadv_burgers(hh2_bar) 
        hh_new = hh_avg1 - 0.5 * a*dt/dx * hh_avg2           # half step

        hh_new = np.pad(hh_new[bnd_limits[0]:-bnd_limits[1]],bnd_limits,bnd_type)
        uunt[i] = hh_new      
        hh = hh_new
        tt[i] = tt[i-1] + dt2

    return tt, uunt


def osp_Lax_LH_Strang(xx, hh, nt, a, b, cfl_cut = 0.98, 
        ddx = lambda x,y: deriv_dnw(x, y), 
        bnd_type='wrap', bnd_limits=[0,1], **kwargs): 
    r"""
    Advance nt time-steps in time the burger eq for a being a and b 
    a fix constant or array. Solving two advective terms separately 
    with the Strang Operator Splitting scheme. One step is with a Lax method 
    and the second step is the Hyman predictor-corrector scheme. 

    Requires
    --------
    step_adv_burgers
    cfl_adv_burger

    Parameters
    ----------
    xx : `array`
        Spatial axis. 
    hh : `array`
        Function that depends on xx.
    nt : `int`
        Number of iterations.
    a : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    b : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    cfl_cut : `float` 
        Limit dt from cfl_adv_burger. 
        By default 0.98
    ddx : `lambda function` 
        Allows to change the space derivative function. 
        By default lambda x,y: deriv_dnw(x, y)
    bnd_type : `string`
        It allows to select the type of boundaries. 
        By default 'wrap'
    bnd_limits : `list(int)`
        Array of two integer elements. The number of pixels that
        will need to be updated with the boundary information. 
        By default [0,1]

    Returns
    -------
    t : `array`
        Time 1D array
    unnt : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain. 
    """


def step_diff_burgers(xx, hh, a=1, ddx = lambda x,y: deriv_cent(x, y), **kwargs): 
    r"""
    Right hand side of the diffusive term of Burger's eq. where nu can be a constant or a function that 
    depends on xx. 
    
    Parameters
    ----------    
    xx : `array`
        Spatial axis. 
    hh : `array`
        Function that depends on xx.
    a : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    ddx : `lambda function`
        Allows to change the space derivative function. 
        By default lambda x,y: deriv_dnw(x, y)

    Returns
    -------
    `array`
        Right hand side of (u^{n+1}-u^{n})/dt = from burgers eq, i.e., x \frac{\partial u}{\partial x} 
    """

    dx = xx[1]-xx[0]

    step = a/(dx*dx)*(np.roll(hh,1) - 2*hh + np.roll(hh,-1))
    return step





def NR_f(xx, un, uo, a, dt, **kwargs): 
    r"""
    NR F function. 

    Parameters
    ----------   
    xx : `array`
        Spatial axis. 
    un : `array`
        Function that depends on xx.
    uo : `array`
        Function that depends on xx.
    a : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    dt : `float` 
        Time interval

    Returns
    -------
    `array`
        function  u^{n+1}_{j}-u^{n}_{j} - a (u^{n+1}_{j+1} - 2 u^{n+1}_{j} -u^{n+1}_{j-1}) dt
    """    

    #F = np.zeros(un.shape)
    dx = xx[1] - xx[0]
    alpha = a*dt / (dx*dx)

    F = un - uo - alpha * (np.roll(un,1) - 2*un + np.roll(un,-1))
    return F



def jacobian(xx, un, a, dt, **kwargs): 
    r"""
    Jacobian of the F function. 

    Parameters
    ----------   
    xx : `array`
        Spatial axis. 
    un : `array`
        Function that depends on xx.
    a : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    dt : `float` 
        Time interval

    Returns
    -------
    `array`
        Jacobian F_j'(u^{n+1}{k})
    """   

    

    dx = xx[1] - xx[0]
    #dx = 1
    lenx = un.shape[0]
    J = np.zeros((lenx,lenx))
    diag = 1+2*a*dt/(dx*dx)
    seconddiag = -a*dt/(dx*dx)
    maxshape = min(un.shape)
    for i in range(0,maxshape):
        J[i][i] = diag
    
    for i in range(1,maxshape-1):
        J[i][i+1] = seconddiag
        J[i][i-1] = seconddiag

    J[0][1] = seconddiag
    J[maxshape-1][maxshape-2] = seconddiag 

    return J

    


    


def Newton_Raphson(xx, hh, a, dt, nt, toll= 1e-5, ncount=2, 
            bnd_type='wrap', bnd_limits=[1,1], **kwargs):
    r"""
    NR scheme for the burgers equation. 

    Parameters
    ----------   
    xx : `array`
        Spatial axis. 
    hh : `array`
        Function that depends on xx.
    a : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    dt : `float`
        Time interval
    nt : `int`
        Number of iterations
    toll : `float` 
        Error limit.
        By default 1e-5
    ncount : `int`
        Maximum number of iterations.
        By default 2
    bnd_type : `string` 
        Allows to select the type of boundaries.
        By default 'wrap'
    bnd_limits : `list(int)`
        Array of two integer elements. The number of pixels that
        will need to be updated with the boundary information.
        By default [1,1]

    Returns
    -------
    t : `array`
        Array of time. 
    unnt : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain. 
    errt : `array`
        Error for each timestep
    countt : `list(int)`
        number iterations for each timestep
    """    
    err=1.
    unnt = np.zeros((np.size(xx),nt))
    errt = np.zeros((nt))
    countt = np.zeros((nt))
    unnt[:,0] = hh
    t=np.zeros((nt))
    
    ## Looping over time 
    for it in range(1,nt): 
        uo=unnt[:,it-1]
        ug=unnt[:,it-1] 
        count = 0 
        # iteration to reduce the error. 
        while ((err >= toll) and (count < ncount)): 

            jac = jacobian(xx, ug, a, dt) # Jacobian 
            ff1=NR_f(xx, ug, uo, a, dt) # F 
            # Inversion: 
            un = ug - np.matmul(np.linalg.inv(
                    jac),ff1)

            # error: 
            err = np.max(np.abs(un-ug)/(np.abs(un)+toll)) # error
            #err = np.max(np.abs(un-ug))
            errt[it]=err

            # Number of iterations
            count+=1
            countt[it]=count
            
            # Boundaries 
            if bnd_limits[1]>0: 
                u1_c = un[bnd_limits[0]:-bnd_limits[1]]
            else: 
                u1_c = un[bnd_limits[0]:]
            un = np.pad(u1_c, bnd_limits, bnd_type)
            ug = un 
        err=1.
        t[it] = t[it-1] + dt
        unnt[:,it] = un
        
    return t, unnt, errt, countt



def NR_f_u(xx, un, uo, dt, **kwargs): 
    r"""
    NR F function.

    Parameters
    ----------  
    xx : `array`
        Spatial axis. 
    un : `array`
        Function that depends on xx.
    uo : `array`
        Function that depends on xx.
    a : `float` and `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    dt : `int`
        Time interval

    Returns
    -------
    `array`
        function  u^{n+1}_{j}-u^{n}_{j} - a (u^{n+1}_{j+1} - 2 u^{n+1}_{j} -u^{n+1}_{j-1}) dt
    """    

    dx = xx[1] - xx[0]
    alpha = un*dt/(dx*dx)

    F = un - uo - alpha * (np.roll(un,1) - 2*un + np.roll(un,-1))
    return F


def jacobian_u(xx, un, dt, **kwargs): 
    """
    Jacobian of the F function. 

    Parameters
    ----------   
    xx : `array`
        Spatial axis. 
    un : `array`
        Function that depends on xx.
    a : `float` and `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    dt : `int`
        Time interval

    Returns
    -------
    `array`
        Jacobian F_j'(u^{n+1}{k})
    """

    dx = xx[1] - xx[0]
    lenx = un.shape[0]
    J = np.zeros((lenx,lenx))
    diag = 1+2*un*dt/(dx*dx)
    seconddiagp = -np.roll(un,1)*dt/(dx*dx)
    seconddiagm = -np.roll(un,-1)*dt/(dx*dx)
    maxshape = min(un.shape)
    for i in range(0,maxshape):
        J[i][i] = diag[i]
    
    for i in range(1,maxshape-1):
        J[i][i+1] = seconddiagp[i+1]
        J[i][i-1] = seconddiagm[i-1]

    J[0][1] = seconddiagp[1]
    J[maxshape-1][maxshape-2] = seconddiagm[-1]

    return J    


def deriv_Lax_Wendroff(xx,hh,dt,lmd = 1):

    dx = xx[1] - xx[-1]

    return -0.5*lmd*(np.roll(hh,1)/dx - np.roll(hh,-1)) + 0.5*dt*lmd*(np.roll(hh,1)-2*hh+np.roll(hh,-1))/(dx*dx)





def Newton_Raphson_u(xx, hh, dt, nt, toll= 1e-5, ncount=2, 
            bnd_type='wrap', bnd_limits=[1,1], **kwargs):
    """
    NR scheme for the burgers equation. 

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    hh : `array`
        Function that depends on xx.
    dt : `float` 
        Time interval
    nt : `int`
        Number of iterations
    toll : `float` 
        Error limit.
        By default 1-5
    ncount : `int`
        Maximum number of iterations.
        By default 2
    bnd_type : `string` 
        Allows to select the type of boundaries.
        By default 'wrap'
    bnd_limits : `list(int)`
        Array of two integer elements. The number of pixels that
        will need to be updated with the boundary information.
        By default [1,1]        

    Returns
    -------
    t : `array`
        Time. 
    unnt : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain. 
    errt : `array`
        Error for each timestep
    countt : `array(int)` 
        Number iterations for each timestep
    """    
    err=1.
    unnt = np.zeros((np.size(xx),nt))
    errt = np.zeros((nt))
    countt = np.zeros((nt))
    unnt[:,0] = hh
    t=np.zeros((nt))
    
    ## Looping over time 
    for it in range(1,nt): 
        uo=unnt[:,it-1]
        ug=unnt[:,it-1] 
        count = 0 
        # iteration to reduce the error. 
        while ((err >= toll) and (count < ncount)): 

            jac = jacobian_u(xx, ug, dt) # Jacobian 
            ff1=NR_f_u(xx, ug, uo, dt) # F 
            # Inversion: 
            un = ug - np.matmul(np.linalg.inv(
                    jac),ff1)

            # error
            err = np.max(np.abs(un-ug)/(np.abs(un)+toll)) 
            errt[it]=err

            # Number of iterations
            count+=1
            countt[it]=count
            
            # Boundaries 
            if bnd_limits[1]>0: 
                u1_c = un[bnd_limits[0]:-bnd_limits[1]]
            else: 
                u1_c = un[bnd_limits[0]:]
            un = np.pad(u1_c, bnd_limits, bnd_type)
            ug = un 
        err=1.
        t[it] = t[it-1] + dt
        unnt[:,it] = un
        
    return t, unnt, errt, countt

def taui_sts(nu, niter, iiter): 
    """
    STS parabolic scheme. [(nu -1)cos(pi (2 iiter - 1) / 2 niter) + nu + 1]^{-1}

    Parameters
    ----------   
    nu : `float`
        Coefficient, between (0,1).
    niter : `int` 
        Number of iterations
    iiter : `int`
        Iterations number

    Returns
    -------
    `float` 
        [(nu -1)cos(pi (2 iiter - 1) / 2 niter) + nu + 1]^{-1}
    """


    return 1/((nu-1) * np.cos(np.pi * (2*iiter - 1) / (2*niter)) + nu + 1)





def evolv_sts(xx, hh, nt,  a, cfl_cut = 0.45, 
        ddx = lambda x,y: deriv_cent(x, y), 
        bnd_type='wrap', bnd_limits=[0,1], nu=0.9, n_sts=10): 
    """
    Evolution of the STS method. 

    Parameters
    ----------
    xx : `array`
        Spatial axis. 
    hh : `array`
        Function that depends on xx.
    nt : `int`
        Number of iterations
    a : `float` or `array` 
        Either constant, or array which multiply the right hand side of the Burger's eq.
    cfl_cut : `float`
        Constant value to limit dt from cfl_adv_burger. 
        By default 0.45
    ddx : `lambda function` 
        Allows to change the space derivative function. 
        By default lambda x,y: deriv_cent(x, y)
    bnd_type : `string` 
        Allows to select the type of boundaries
        by default 'wrap'
    bnd_limits : `list(int)`
        List of two integer elements. The number of pixels that
        will need to be updated with the boundary information. 
        By defalt [0,1]
    nu : `float`
        STS nu coefficient between (0,1).
        By default 0.9
    n_sts : `int`
        Number of STS sub iterations. 
        By default 10

    Returns
    -------
    t : `array`
        time 1D array
    unnt : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain. 
    """


    tt = np.zeros(nt)
    unnt = np.zeros((nt,xx.shape[0]))
    dx = xx[1] - xx[0]

    #Initial condition
    unnt[0,:] = hh.copy()
    
    #Cfl condition
    dt_cfl = cfl_cut * cfl_diff_burger(a,xx)
    t = 0
    for i in range(1,nt):
        dt = 0

        for sts_iter in range(1,n_sts+1):
            rhs = step_diff_burgers(xx,hh,a, ddx=ddx)
            #substep
            tau = taui_sts(nu,n_sts,sts_iter)
            
            hh = hh + rhs*tau*dt_cfl
            dt += tau

            #Boundary
            hh = np.pad(hh[bnd_limits[0]:-bnd_limits[1]],bnd_limits,bnd_type)


        unnt[i,:] = hh.copy()
        t += dt*dt_cfl
        tt[i] = t


    return tt, unnt



def hyman(xx, f, dth, a, fold=None, dtold=None,
        cfl_cut=0.8, ddx = lambda x,y: deriv_dnw(x, y), 
        bnd_type='wrap', bnd_limits=[0,1], **kwargs): 

    dt, u1_temp = step_adv_burgers(xx, f, a, ddx=ddx)

    if (np.any(fold) == None):
        firstit=False
        fold = np.copy(f)
        f = (np.roll(f,1)+np.roll(f,-1))/2.0 + u1_temp * dth 
        dtold=dth

    else:
        ratio = dth/dtold
        a1 = ratio**2
        b1 =  dth*(1.0+ratio   )
        a2 =  2.*(1.0+ratio    )/(2.0+3.0*ratio)
        b2 =  dth*(1.0+ratio**2)/(2.0+3.0*ratio)
        c2 =  dth*(1.0+ratio   )/(2.0+3.0*ratio)

        f, fold, fsav = hyman_pred(f, fold, u1_temp, a1, b1, a2, b2)
        
        if bnd_limits[1]>0: 
            u1_c =  f[bnd_limits[0]:-bnd_limits[1]]
        else: 
            u1_c = f[bnd_limits[0]:]
        f = np.pad(u1_c, bnd_limits, bnd_type)

        dt, u1_temp = step_adv_burgers(xx, f, a, cfl_cut, ddx=ddx)

        f = hyman_corr(f, fsav, u1_temp, c2)

    if bnd_limits[1]>0: 
        u1_c = f[bnd_limits[0]:-bnd_limits[1]]
    else: 
        u1_c = f[bnd_limits[0]:]
    f = np.pad(u1_c, bnd_limits, bnd_type)
    
    dtold=dth

    return f, fold, dtold


def hyman_corr(f, fsav, dfdt, c2):

    return  fsav  + c2* dfdt


def hyman_pred(f, fold, dfdt, a1, b1, a2, b2): 

    fsav = np.copy(f)
    tempvar = f + a1*(fold-f) + b1*dfdt
    fold = np.copy(fsav)
    fsav = tempvar + a2*(fsav-tempvar) + b2*dfdt    
    f = tempvar
    
    return f, fold, fsav

