import numpy as np

def cfx_cal(cfx,nx,un,hs_up,g,snm):
    for i in np.arange(0,nx+1):
        cfx[i]=-g*snm**2*un[i]*np.abs(un[i])/hs_up[i]**(4./3.)
    return cfx
