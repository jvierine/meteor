#!/usr/bin/env python3
import matplotlib
#matplotlib.use('Agg')
import numpy as n
import matplotlib.pyplot as plt
import glob
import scipy.interpolate as sint
import itertools
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

#
# From Toralf
#
#Phases CasA Observations 
#Ant - No. of data - Median Phase /degr - Standard Dev /degr - Median Phase /rad - Standard Dev /rad   
#433 3338 160.706296 10.366737 2.804854 0.180934 
#A 3338 17.715985 22.439305 0.309202 0.391640 
#B 3338 5.499101 12.702822 0.095977 0.221706 
#C 3338 0.000000 0.000000 0.000000 0.000000 
#D 3338 -5.865708 8.600339 -0.102376 0.150104 
#E 3338 -12.842389 17.223328 -0.224142 0.300604 
#F 3338 4.229617 26.690570 0.073821 0.465838 
#M 3338 -8.033325 16.044955 -0.140208 0.280037 
#A-07 2051 80.812199 30.860553 1.410439 0.538618 
#B-06 3338 76.720319 11.094380 1.339022 0.193633 
#C-02 3338 82.305434 8.910438 1.436501 0.155516 
#D-06 3338 59.669542 15.508242 1.041430 0.270670 
#E-07 3338 48.753974 26.552948 0.850917 0.463436 
#X-02 3338 108.650792 39.489563 1.896314 0.689223 
#F-09 3338 60.193235 22.292532 1.050570 0.389078 
#B-08 3338 69.198433 18.024550 1.207740 0.314588 
#A-01 1287 123.543139 23.352234 2.156235 0.407573 

phases_deg=n.array([160.706296, # 433
                    17.715985,  # A
                    5.499101,   # B
                    0,          # B
                    -5.865708,  # C
                    -12.842389, # D
                    4.229617,   # E
                    -8.033325,  # F
                    123.543139, # A-01
                    76.720319,  # B-06
                    82.305434,  # C-02
                    59.669542,  # D-06
                    48.753974,  # E-07
                    108.650792, # X-02
                    60.193235,  # F-09
                    69.198433   # B08
                    ],dtype=n.float64)
phases_rad=n.pi*phases_deg/180.0

def get_codes():
    code_strings=["1000110110000010","0100000101001110"]
    codes=[]
    code=[]
    for i in range(16):
        a=int(code_strings[0][i])
        if(a == 0):
            a=-1.0
        else:
            a=1.0
        code.append(a)
    codes.append(code)
    code=[]
    for i in range(16):
        a=int(code_strings[1][i])
        if(a == 0):
            a=-1.0
        else:
            a=1.0
        code.append(a)
    codes.append(code)
    return(codes)

def read_ud3(fname="GEMINIDS/20221212_023647112_event.ud3"):
    f=open(fname,"rb")
    l=f.readline()
    cont=True
    M_DATAPOINTS=0
    M_GATES=0
    M_RANGE=0.0
    while cont:
        #        print(l[0:5])
        if l[0:5]==b'DATA ':
            cont=False
        if l[0:5] != b'DATA ':
            print("key")
            print(l)
            if l[0:7] == b'M_GATES':
                M_GATES=int(l[8:11])
            if l[0:7] == b'M_RANGE':                
                M_RANGE=int(l[8:14])
            print(l[0:12])
            if l[0:12] == b'M_DATAPOINTS':
                M_DATAPOINTS=int(l[13:17])
#                print(l)

        if cont:
            l=f.readline()[:-1]
    z=n.fromfile(f,dtype=n.int16)
    z2=n.array(z[0:len(z):2]+z[1:len(z):2]*1j,dtype=n.complex64)
    z2.shape=(M_DATAPOINTS*2,M_GATES,16)

    # phase cal
    for i in range(16):
        z2[:,:,i]=z2[:,:,i]*n.exp(-1j*phases_rad[i])
        

    
    print(len(z))
    print(M_DATAPOINTS)
    print(M_GATES)
    print(M_DATAPOINTS*2*M_GATES*2*16)
#    z.shape=(16,2,M_DATAPOINTS*2,M_GATES) not this
#    z.shape=(M_DATAPOINTS*2,16,2,M_GATES) not this
#    z.shape=(M_DATAPOINTS*2,M_GATES,16,2) not this
   # z2.shape=(M_DATAPOINTS*2,M_GATES,16)# not this
    if False:
        plt.pcolormesh(n.transpose(n.real(z2[:,:,0])))
        plt.colorbar()
        plt.show()
    f.close()
    return(z2,{"m_range":M_RANGE})
                   

def antenna_pos():
    ant_str="0.00 0.00 28.00 15.00 28.00 75.00 28.00 135.00 28.00 195.00 28.00 255.00 28.00 315.00 0.00 0.00 38.16 9.79 28.00 96.79 31.75 115.89 28.00 216.79 36.66 265.89 67.01 137.89 10.58 295.89 38.16 102.00"
    aa=n.array(ant_str.split(" "),dtype=n.float64)
    radius=aa[0:32:2]
    angle=n.pi*aa[1:32:2]/180.0
    # x is east-west
    x=radius*n.sin(angle)
    # y is north-south
    y=radius*n.cos(angle)
    return(x,y)

def uv_coverage(x,y,pairs,N=1000):
    u=n.zeros(len(pairs))
    v=n.zeros(len(pairs))
    for pi in range(len(pairs)):
        u[pi]=x[pairs[pi][0]]-x[pairs[pi][1]]
        v[pi]=y[pairs[pi][0]]-y[pairs[pi][1]]

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



def analyze_file(f,N=100):

    # interpolate codes
    codes=get_codes()

    # read data file
    z,o=read_ud3(fname=f)

    pairs=list(itertools.combinations(n.arange(16,dtype=int),2))
    
    n_xspec=len(pairs)
    S=n.zeros([z.shape[0],z.shape[1],n_xspec],dtype=n.complex64)

    # self correlations
    C=n.zeros([z.shape[0],z.shape[1],16,2],dtype=n.float64)
    for j in range(16):
        for i in range(int(z.shape[0]/2)):
            # not aliased. 
            cc0=n.fft.ifft(n.fft.fft(z[2*i,:,j],z.shape[1])*n.conj(n.fft.fft(codes[0],z.shape[1])))
            C[2*i,:,j,0]=n.real(cc0*n.conj(cc0))
            cc0=n.fft.ifft(n.fft.fft(z[2*i+1,:,j],z.shape[1])*n.conj(n.fft.fft(codes[1],z.shape[1])))
            C[2*i+1,:,j,0]=n.real(cc0*n.conj(cc0))
            # range aliased
            cc0=n.fft.ifft(n.fft.fft(z[2*i,:,j],z.shape[1])*n.conj(n.fft.fft(codes[1],z.shape[1])))
            C[2*i,:,j,1]=n.real(cc0*n.conj(cc0))
            cc0=n.fft.ifft(n.fft.fft(z[2*i+1,:,j],z.shape[1])*n.conj(n.fft.fft(codes[0],z.shape[1])))
            C[2*i+1,:,j,1]=n.real(cc0*n.conj(cc0))
            
    S0=n.sum(n.abs(C[:,:,:,0]),axis=2)
    S1=n.sum(n.abs(C[:,:,:,1]),axis=2)
    pwr0=n.max(S0,axis=1)
    pwr1=n.max(S1,axis=1)    

    aliased=False
    if n.sum(pwr1) > n.sum(pwr0):
        aliased=True

    if aliased:
        P=S1
        C=C[:,:,0,1]
    else:
        C=C[:,:,0,0]
        P=S0
        
        
    if aliased:
        plt.pcolormesh(n.transpose(10.0*n.log10(P)))
        plt.show()
    else:
        plt.pcolormesh(n.transpose(10.0*n.log10(P)))
        plt.show()
        

    x,y=antenna_pos()
    u,v,ui,vi=uv_coverage(x,y,pairs,N=N)
    
    #S=n.copy(z)
    #S[:,:,:]=0.0
    for pi in range(len(pairs)):
        pidx=pairs[pi]
        for i in range(int(z.shape[0]/2)):
            idx0=pidx[0]
            idx1=pidx[1]
            code0=0
            code1=1
            if aliased:
                code0=1
                code1=0
        
            cc0=n.fft.ifft(n.fft.fft(z[2*i,:,idx0],z.shape[1])*n.conj(n.fft.fft(codes[code0],z.shape[1])))
            cc1=n.fft.ifft(n.fft.fft(z[2*i,:,idx1],z.shape[1])*n.conj(n.fft.fft(codes[code0],z.shape[1])))
            
            S[2*i,:,pi]=cc0*n.conj(cc1)
            S[2*i,:,pi]=S[2*i,:,pi]
            
            cc0=n.fft.ifft(n.fft.fft(z[2*i+1,:,idx0],z.shape[1])*n.conj(n.fft.fft(codes[code1],z.shape[1])))
            cc1=n.fft.ifft(n.fft.fft(z[2*i+1,:,idx1],z.shape[1])*n.conj(n.fft.fft(codes[code1],z.shape[1])))

            S[2*i+1,:,pi]=cc0*n.conj(cc1)
            S[2*i+1,:,pi]=S[2*i+1,:,pi]
            

    for i in range(S.shape[0]):
        rg=n.argmax(P[i,:])
        
        uidx=n.argsort(u)
        vidx=n.argsort(v)
        print(rg)
        plt.subplot(121)
        plt.plot(u[uidx],n.angle(S[i,rg,uidx]),".")
        plt.subplot(122)
        plt.plot(v[vidx],n.angle(S[i,rg,vidx]),".")
        plt.show()

        
    tm=n.arange(S.shape[0])*1e-3
    rgs=(n.arange(S.shape[1])*300.0 + o["m_range"])/1e3
    #    plt.subplot(121)
    if aliased:
        dB=10.0*n.log10(n.transpose(S1))
        dB=dB-n.median(dB)
        plt.pcolormesh(tm,rgs+0.3*1000/2.0,dB,vmin=-6,vmax=30)
    else:
        dB=10.0*n.log10(n.transpose(S0))
        dB=dB-n.median(dB)
        plt.pcolormesh(tm,rgs,dB,vmin=-6,vmax=30)
    plt.colorbar()
    plt.title(f)
    plt.xlabel("Time (s)")
    plt.ylabel("Range (km)")
    plt.tight_layout()
#    plt.show()
    plt.savefig("tmp/%s.png"%(f))
    plt.clf()
    plt.close()
    del S
    del z
    del o
    #        plt.show()



analyze_file("/data1/geminids/maarsy/GEMINIDS/20221212_023647112_event.ud3")

if __name__ == "__main__":
    fl=glob.glob("/data1/geminids/maarsy/GEMINIDS/2*.ud3")
    fl.sort()
    for fi in range(rank, len(fl), size):
        f=fl[fi]
        analyze_file(f)



