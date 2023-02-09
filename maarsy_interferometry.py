import numpy as n
import matplotlib.pyplot as plt

import maarsy_config as mc

def uv_coverage(x,y,pairs,N=1000):
        
    u=n.zeros(len(pairs))
    v=n.zeros(len(pairs))
    for pi in range(len(pairs)):
        u[pi]=x[pairs[pi][1]]-x[pairs[pi][0]]
        v[pi]=y[pairs[pi][1]]-y[pairs[pi][0]]

    urange=2*n.max([n.abs(n.max(u)),n.abs(n.min(u))])
    vrange=2*n.max([n.abs(n.max(v)),n.abs(n.min(v))])    
    uvrange=n.max([urange,vrange])
    du=uvrange/N
    uidx=n.array(n.round(u/du),dtype=int)
    vidx=n.array(n.round(v/du),dtype=int)
    uidx[uidx<0]=N+uidx[uidx<0]
    vidx[vidx<0]=N+vidx[vidx<0]    
 #   plt.plot(uidx,vidx,"x")
  #  plt.show()

    return(u,v,uidx,vidx)

def kvecs(N=400,maxdcos=0.3,k=2.0*n.pi/mc.lam):
    l=n.linspace(-maxdcos,maxdcos,num=N)
    m=n.linspace(-maxdcos,maxdcos,num=N)    
    ll,mm=n.meshgrid(l,m)
    nn=n.sqrt(1-ll**2.0+mm**2.0)
    kvec_x = k*ll
    kvec_y = k*mm
    kvec_z = k*nn
    mask = n.sqrt(ll**2.0+mm**2.0) < maxdcos
    return(kvec_x,kvec_y,l,m,mask)

def find_angle(u,v,S,kvec_x,kvec_y,l,m,weights=[],mask=1.0):
    if len(weights)==0:
        weights=n.repeat(1.0,len(u))

    meas = n.exp(1j*n.angle(S))

    MF = n.zeros(kvec_x.shape,dtype=n.complex64)
    for i in range(len(meas)):
        MF+=meas[i]*n.exp(-1j*(kvec_x*u[i] + kvec_y*v[i]))*weights[i]
    MF=MF*mask
        
    i,j=n.unravel_index(n.argmax(n.abs(MF)),kvec_x.shape)
    if False:
        plt.pcolormesh(n.abs(MF))
        plt.colorbar()
        plt.show()
    return(l[i],m[j])

if __name__ == "__main__":
    x,y=mc.antenna_pos()
    u,v,ui,vi=uv_coverage(x,y,mc.pairs,N=100)
    plt.plot(x,y,".")
    plt.show()
    
    plt.plot(u,v,".")
    plt.show()
