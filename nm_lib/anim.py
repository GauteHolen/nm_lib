"""
Animation tools

"""


from cmd import IDENTCHARS
from re import I
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# import external public "common" modules
import numpy as np
import matplotlib.pyplot as plt 


def animMultTsync(uts,xx,lbls, styles, ts,n_frames=100, nt = False, log_time = False,ylim = None):
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
            axes.set_title('t=%.5f'%t_ref[i])
        plt.legend()

    t_ref = ts[0]
    for i in range(len(uts)):
        t = ts[i]
        ut = uts[i]
        ts[i],uts[i] = get_closest_ut(t_ref,t,ut)

    Nt = len(t_ref)

    
    if n_frames > len(t_ref):
        print("WARNING: number of frames exceeds number of timesteps")
        n_frames = len(t_ref)

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




def animMHDparams(et,ut,rhot,Pt,t,xx,n_frames=100, nt = False, log_time = False,ylim = None):
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
        axes[0][0].plot(xx,rhot[0])
        axes[0][0].set_title('$\rho$')
        axes[0][1].plot(xx,Pt[0])
        axes[0][1].set_title('$P$')
        axes[1][0].plot(xx,et[0])
        axes[1][0].set_title('$e$')
        axes[1][1].plot(xx,ut[0])
        axes[1][1].set_title('$u$')

        if ylim:
            axes[0][0].set_ylim(ylim)
            axes[0][1].set_ylim(ylim)
            axes[1][0].set_ylim(ylim)
            axes[1][1].set_ylim((-ylim[1],ylim[1]))


        fig.suptitle('t=%.6f'%t_ref[0])

    def animate(i):
        for ax1 in axes:
            for ax in ax1:
                ax.clear()

        axes[0][0].plot(xx,rhot[i])
        axes[0][0].set_title(r'$\rho$')
        axes[0][1].plot(xx,Pt[i])
        axes[0][1].set_title('$P$')
        axes[1][0].plot(xx,et[i])
        axes[1][0].set_title('$e$')
        axes[1][1].plot(xx,ut[i])
        axes[1][1].set_title('$u$')

        if ylim:
            axes[0][0].set_ylim(ylim)
            axes[0][1].set_ylim(ylim)
            axes[1][0].set_ylim(ylim)
            axes[1][1].set_ylim((-ylim[1],ylim[1]))

        fig.suptitle('t=%.6f'%t_ref[i])
        


    t_ref = t
    Nt = len(t)

    
    if n_frames > len(t_ref):
        print("WARNING: number of frames exceeds number of timesteps")
        n_frames = len(t_ref)

    if log_time:
        frames = np.zeros(n_frames, dtype=np.int64)
        #First some linearly spaced frames
        lin_frames = int(0.20*n_frames)
        frames[0:lin_frames] = np.linspace(0,lin_frames-1,lin_frames, dtype=np.int64)
        #Log spaced frames
        frames[lin_frames:] = np.geomspace(lin_frames,Nt-1,num=n_frames-lin_frames,dtype=np.int64)[:]
    else:
        frames = np.linspace(0,Nt-1,num=n_frames, dtype=np.int64)

    
    
    fig, axes = plt.subplots(2,2, figsize=(10, 5), tight_layout = True)
    anim = FuncAnimation(fig, animate, interval=20, frames=frames, init_func=init)
    html = HTML(anim.to_jshtml())
    plt.close()
    return html


def animMultMHDparams(names,ets,uts,rhots,Pts,t,xx,n_frames=100, nt = False, log_time = False,ylim = None):
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
        for rhot,Pt,et,ut,name in zip(rhots,Pts,ets,uts,names):
            axes[0][0].plot(xx,rhot[0],label = name)
            axes[0][0].set_title('$\rho$')
            axes[0][1].plot(xx,Pt[0],label = name)
            axes[0][1].set_title('$P$')
            axes[1][0].plot(xx,et[0],label = name)
            axes[1][0].set_title('$e$')
            axes[1][1].plot(xx,ut[0],label = name)
            axes[1][1].set_title('$u$')
            if ylim:
                axes[0][0].set_ylim(ylim)
                axes[0][1].set_ylim(ylim)
                axes[1][0].set_ylim(ylim)
                axes[1][1].set_ylim((-ylim[1],ylim[1]))
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper left')
        fig.suptitle('t=%.6f'%t_ref[0])

    def animate(i):
        for ax1 in axes:
            for ax in ax1:
                ax.clear()
        for rhot,Pt,et,ut,name in zip(rhots,Pts,ets,uts,names):
            if ylim:
                axes[0][0].set_ylim(ylim)
                axes[0][1].set_ylim(ylim)
                axes[1][0].set_ylim(ylim)
                axes[1][1].set_ylim((-ylim[1],ylim[1]))
                #if np.amax(np.abs(et[i])) > 5*ylim[1]:
                """
                if np.amin(Pt[i]).all() > 0:
                    #print(i,np.amax(np.abs(et[i])),5*ylim[1])
                    axes[0][0].plot(xx,rhot[i],label = name)
                    axes[0][0].set_title(r'$\rho$')
                    axes[0][1].plot(xx,Pt[i],label = name)
                    axes[0][1].set_title('$P$')
                    axes[1][0].plot(xx,et[i],label = name)
                    axes[1][0].set_title('$e$')
                    axes[1][1].plot(xx,ut[i],label = name)
                    axes[1][1].set_title('$u$')
                """
            #else:
            #if np.amax(np.abs(et[i])) > 5*ylim[1]:
            #    name="Exploded"
            axes[0][0].plot(xx,rhot[i],label = name)
            axes[0][0].set_title(r'$\rho$')
            axes[0][1].plot(xx,Pt[i],label = name)
            axes[0][1].set_title('$P$')
            axes[1][0].plot(xx,et[i],label = name)
            axes[1][0].set_title('$e$')
            axes[1][1].plot(xx,ut[i],label = name)
            axes[1][1].set_title('$u$')
            
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper left')
        fig.suptitle('t=%.6f'%t_ref[i])
        


    t_ref = t
    Nt = len(t)

    
    if n_frames > len(t_ref):
        print("WARNING: number of frames exceeds number of timesteps")
        n_frames = len(t_ref)

    if log_time:
        frames = np.zeros(n_frames, dtype=np.int64)
        #First some linearly spaced frames
        lin_frames = int(0.20*n_frames)
        frames[0:lin_frames] = np.linspace(0,lin_frames-1,lin_frames, dtype=np.int64)
        #Log spaced frames
        frames[lin_frames:] = np.geomspace(lin_frames,Nt-1,num=n_frames-lin_frames,dtype=np.int64)[:]
    else:
        frames = np.linspace(0,Nt-1,num=n_frames, dtype=np.int64)

    
    
    fig, axes = plt.subplots(2,2, figsize=(15, 10), tight_layout = True)
    anim = FuncAnimation(fig, animate, interval=20, frames=frames, init_func=init)
    html = HTML(anim.to_jshtml())
    plt.close()
    return html


def get_closest_ut(t_ref,tt,ut):
    """Gets the closest t and u values from a given t_ref. Used to sync
    up solutions with different timesteps for animations


    Args:
        t_ref (arr): time array to be synced to
        tt (arr): time array to be synced
        ut (arr): corresponding spacial acis to be synced

    Returns:
        t_closest (arr): closest values of tt to t_ref
        u_closest (arr): corresponding u values of t_closest
    """

    t_closest = np.zeros(len(t_ref))
    u_closest = np.zeros(ut.shape)
    for i,t in enumerate(t_ref):
        intersect = np.abs(tt-t)
        idx = np.argmin(intersect)
        #idx = idx[0]
        t_closest[i] = tt[idx]
        u_closest[i] = ut[idx]


    return t_closest,u_closest



def unpack_CFD_values(values):
    names = []
    uts = []
    rhots = []
    Pts = []
    ets = []
    for value in values:
        [name,ut,rhot,Pt,et] = value
        names.append(name)
        uts.append(ut)
        rhots.append(rhot)
        Pts.append(Pt)
        ets.append(et)
    return names,uts,rhots,Pts,ets