#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')
import numpy as n
import matplotlib.pyplot as plt
import glob
import scipy.interpolate as sint
import itertools
from mpi4py import MPI
import scipy.constants as c
import stuffr
import h5py
import maarsy_config as mc

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
#lam=c.c/53.5e6
#phases_deg=n.array([160.706296, # 433
 #                   17.715985,  # A
  #                  5.499101,   # B
   #                 0,          # C
    #                -5.865708,  # D
     #               -12.842389, # E
      #              4.229617,   # F
       #             -8.033325,  # M
        #            123.543139, # A-01
         #           76.720319,  # B-06
          #          82.305434,  # C-02
           #         59.669542,  # D-06
            #        48.753974,  # E-07
             #       108.650792, # X-02
              #      60.193235,  # F-09
               #     69.198433   # B08
                #    ],dtype=n.float64)

#phases_rad=n.pi*phases_deg/180.0

#pairs=[(9,15),(9,10),(10,15),
#       (7,1),(4,7),(6,5),(2,3),
#       (7,2),(6,1),(5,7),(4,3),
#       (7,3),(1,2),(6,7),(5,4)]

#other_weight=0.25
#weights=[1,1,1,
#         other_weight/4, other_weight/4, other_weight/4.0, other_weight/4, # 2-8 alignment
#         other_weight/4, other_weight/4, other_weight/4.0, other_weight/4, # 8-3 alignment
#         other_weight/4, other_weight/4, other_weight/4.0, other_weight/4] # 8-4 alignment         


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
    M_TIMEOFFSET=0
    t0=0
    while cont:
        #        print(l[0:5])
        if l[0:5]==b'DATA ':
            cont=False
            t0=int(l[5:16])
        if l[0:5] != b'DATA ':
#            print("key")
 #           print(l)
            if l[0:7] == b'M_GATES':
                M_GATES=int(l[8:11])
            if l[0:7] == b'M_RANGE':                
                M_RANGE=int(l[8:14])
#            print(l[0:12])
            if l[0:12] == b'M_DATAPOINTS':
                M_DATAPOINTS=int(l[13:17])
            if l[0:12] == b'M_TIMEOFFSET':
                M_TIMEOFFSET=float(l[13:24])

        if cont:
            l=f.readline()[:-1]
    z=n.fromfile(f,dtype=n.int16)
    z2=n.array(z[0:len(z):2]+z[1:len(z):2]*1j,dtype=n.complex64)
    z2.shape=(M_DATAPOINTS*2,M_GATES,16)

    # phase cal
    for i in range(16):
        z2[:,:,i]=z2[:,:,i]*n.exp(1j*mc.phases_rad[i])
        
#    print(len(z))
#    print(M_DATAPOINTS)
 #   print(M_GATES)
  #  print(M_DATAPOINTS*2*M_GATES*2*16)
#    z.shape=(16,2,M_DATAPOINTS*2,M_GATES) not this
#    z.shape=(M_DATAPOINTS*2,16,2,M_GATES) not this
#    z.shape=(M_DATAPOINTS*2,M_GATES,16,2) not this
   # z2.shape=(M_DATAPOINTS*2,M_GATES,16)# not this
    if False:
        plt.pcolormesh(n.transpose(n.real(z2[:,:,0])))
        plt.colorbar()
        plt.show()
    f.close()
    return(z2,{"m_range":M_RANGE,"t0":t0+M_TIMEOFFSET})

def antenna_pos():
    ant_str="0.00 0.00 28.00 15.00 28.00 75.00 28.00 135.00 28.00 195.00 28.00 255.00 28.00 315.00 0.00 0.00 38.16 9.79 28.00 96.79 31.75 115.89 28.00 216.79 36.66 265.89 67.01 137.89 10.58 295.89 38.16 102.00"
    aa=n.array(ant_str.split(" "),dtype=n.float64)
    radius=aa[0:32:2]
    angle=n.pi*aa[1:32:2]/180.0
    # x is east-west
    x=radius*n.sin(angle)
    # y is north-south
    y=radius*n.cos(angle)

    if False:
        for i in range(16):
            plt.plot(x[i],y[i],"o")
            plt.text(x[i],y[i],"%s"%(i))
        plt.show()

    return(x,y)

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

def kvecs(N=400,maxdcos=0.25,k=2.0*n.pi/mc.lam):
    l=n.linspace(-maxdcos,maxdcos,num=N)
    m=n.linspace(-maxdcos,maxdcos,num=N)    
    ll,mm=n.meshgrid(l,m)
    nn=n.sqrt(1-ll**2.0+mm**2.0)
    kvec_x = k*ll
    kvec_y = k*mm
    kvec_z = k*nn
    return(kvec_x,kvec_y,l,m)

def find_angle(u,v,S,kvec_x,kvec_y,l,m,weights=[]):
    if len(weights)==0:
        weights=n.repeat(1.0,len(u))

    meas = n.exp(1j*n.angle(S))

    MF = n.zeros(kvec_x.shape,dtype=n.complex64)
    for i in range(len(meas)):
        MF+=meas[i]*n.exp(-1j*(kvec_x*u[i] + kvec_y*v[i]))*weights[i]

    i,j=n.unravel_index(n.argmax(n.abs(MF)),kvec_x.shape)
    if False:
        plt.pcolormesh(n.abs(MF))
        plt.colorbar()
        plt.show()
    return(l[i],m[j])


def median_std(S):
    mean_est=n.nanmedian(S)
    std_est=1.48*n.nanmedian(n.abs(S-mean_est))
    return(std_est)

def slots_test(f, snr_limit=25, sr=0.5e6, incoh_dec=16, N=800):
    
    codes=get_codes()
    z,o=read_ud3(fname=f)
    x,y=antenna_pos()
    u,v,ui,vi=uv_coverage(x,y,mc.pairs,N=N)
    kvec_x,kvec_y,lv,mv=kvecs(k=2.0*n.pi/mc.lam,N=N)

    aliased=is_aliased(z,codes)
    
    if aliased:
        print("range aliased echo detected. ignoring")
        return
    
    # for each range
    txlen=16
    nrg=z.shape[1]
    # indices
    idx=n.zeros([nrg,txlen],dtype=int)
    mask=n.zeros([nrg,txlen],dtype=n.complex64)
    mask[:,:]=1.0
    fftfactor=32
    MF=n.zeros([nrg,txlen*fftfactor],dtype=n.float32)
    
    # this is what we select from the echoes
    for ri in range(z.shape[1]):
        idx[ri,:]=n.arange(txlen,dtype=int)+ri

    mask[idx>=z.shape[1]]=0.0
    idx[idx>=z.shape[1]]=z.shape[1]-1

    max_rgs=[]
    max_dops=[]
    max_ss=[]
    ts=[]    
    S=n.zeros([z.shape[0],z.shape[1]],dtype=n.float32)
    D=n.zeros([z.shape[0],z.shape[1]],dtype=n.float32)
    freqvec=n.fft.fftfreq(fftfactor*txlen,d=1.0/sr)
    # 2*fr*v/c=df
    # => df*c/fr/2 = v
    rrvec=c.c*n.fft.fftfreq(fftfactor*txlen,d=1.0/sr)/53.5e6/2.0
    for i in range(int(z.shape[0]-incoh_dec+1)):
        MF[:,:]=0.0
        for k in range(2): # incoh integration. use very little initially!
            for j in range(1):
                MF+=n.abs(n.fft.fft(mask*z[i+k,idx,j]*n.conj(codes[(i+k)%2]),txlen*fftfactor,axis=1))**2.0
        max_rg,max_dop=n.unravel_index(n.argmax(MF),MF.shape)
        S[i,:]=n.max(MF,axis=1)
        D[i,:]=rrvec[n.argmax(MF,axis=1)]
        max_rgs.append(max_rg)
        max_dops.append(max_dop)
        max_ss.append(S[i,max_rg])
        ts.append(i)        

    # figure out approximate range-rate from range-migration
    tmax=z.shape[0]/1e3
    rrmax=200.0
    rrmin=-20.0
    # range-rate step
    drr = 0.15/tmax
    n_rr=int((rrmax-rrmin)/drr)
    rrtest = n.linspace(rrmin,rrmax,num=n_rr)
    tvec=n.arange(z.shape[0])/1e3
    best=0

    std_est=median_std(S)
    
    for i in range(len(rrtest)):
        drs=n.array(n.round(rrtest[i]*tvec/0.3),dtype=int)
        S2=n.copy(S)
#        S2=n.roll(S2,drs,axis=0)
        for ti in range(z.shape[0]):
            S2[ti,:]=n.roll(S2[ti,:],drs[ti])
#            print(int(drs[ti]))
        mf=n.max(n.sum(S2,axis=0))
        if mf > best:
            best=mf
            best_rr=rrtest[i]
            best_s2=S2
            best_drs=drs
            best_rg0=n.argmax(n.sum(best_s2,axis=0))

    line_mf=n.sum(best_s2,axis=0)
    noise_mf=n.median(line_mf)
    peak_snr = n.max(line_mf/noise_mf)

    print("rg0 %d range-rate %1.2f km/s snr %1.2f"%(best_rg0,best_rr,peak_snr))
    
    if n.abs(best_rr) > 4.0 and peak_snr > 1.5:
        
        # only analyze if there is enough doppler velocity
        # unaliased doppler range
        
        if True:
            # unit vectors
            # e-w
            ls=[]
            # n-s
            ms=[]
            # time
            ts=[]
            # power
            ps=[]
            # range
            rgs=[]
            
            n_xspec=len(mc.pairs)
            XC=n.zeros([z.shape[0],z.shape[1],n_xspec],dtype=n.complex64)
            D=n.zeros([z.shape[0],z.shape[1]],dtype=n.complex64)            

            # figure out the range gate
            aliased_rg0 = n.mod(best_rg0,S.shape[1])

            for i in range(len(max_rgs)):
                for pi in range(len(mc.pairs)):
                    pidx=mc.pairs[pi]
                    idx0=pidx[0]
                    idx1=pidx[1]
                    fxc_prev=[]
                    for k in range(incoh_dec):
                        F0=n.fft.fft(mask*z[i+k,idx,idx0]*n.conj(codes[(i+k)%2]),txlen*fftfactor,axis=1)
                        F1=n.fft.fft(mask*z[i+k,idx,idx1]*n.conj(codes[(i+k)%2]),txlen*fftfactor,axis=1)
                        
                        fxc = (F0*n.conj(F1))[:,max_dops[i]]

                        fxc=n.roll(fxc,best_drs[i])
                        # take the max doppler bin of ith time step
                        XC[i,:,pi] += fxc#n.roll(fxc,best_drs[i])

                        
                        if len(fxc_prev) > 0:
                            D[i,:]+=fxc*n.conj(fxc_prev)

                        
                        # previous fxc, used for aliased phase doppler
                        fxc_prev = fxc

                        

            for pi in range(len(mc.pairs)):
                XC[:,:,pi]=XC[:,:,pi]-n.median(XC[:,:,pi])

            P=n.sum(n.abs(XC[:,:,:]),axis=2)
#            plt.pcolormesh(10.0*n.log10(P))
 #           plt.show()
                
            best_rgs=[]
            rg_idx=[aliased_rg0-1,aliased_rg0,aliased_rg0+1]
            for i in range(len(max_rgs)):
                best_rgs.append(rg_idx[n.argmax([P[i,aliased_rg0-1],P[i,aliased_rg0],P[i,aliased_rg0+1]])])
                
            for pi in range(len(mc.pairs)):
                if False:
                    plt.subplot(121)
                    plt.pcolormesh(n.transpose(n.angle(XC[:,:,pi])),cmap="hsv")
                    plt.title(pi)
                    plt.colorbar()
                    plt.subplot(122)
                    plt.pcolormesh(n.transpose(n.abs(XC[:,:,pi])))
                    plt.axhline(aliased_rg0,color="white")
                    plt.title(pi)
                    plt.colorbar()
                    plt.show()
            for pi in range(len(mc.pairs)):
                ph=n.zeros(len(best_rgs),dtype=n.complex64)
                for ti in range(len(best_rgs)):
                    ph[ti]=XC[ti,best_rgs[ti],pi]
                if False:
                    plt.subplot(121)
                    plt.plot(n.angle(ph),".")
                    plt.title(pi)
                    plt.subplot(122)
                    plt.plot(n.abs(ph),".")
                    plt.show()


            # todo: compare to background power of the mean XC...
            threshold=30.0
            
            for i in range(len(max_rgs)):
                if max_ss[i]/std_est > threshold:
                    rgs.append(max_rgs[i]*300 + o["m_range"])
                    l,m=find_angle(u,v,XC[i,best_rgs[i],:],kvec_x,kvec_y,lv,mv,weights=mc.weights)
                    ls.append(l)
                    ms.append(m)
                    ts.append(i*1e-3 + o["t0"])
                    ps.append(max_ss[i])


            plot_3d(ls,ms,ts,rgs,ps,n.arange(XC.shape[0])*1e-3 + o["t0"],(n.arange(XC.shape[1])*300+o["m_range"])/1e3,n.transpose(S)/std_est,o,f)


        # figure out which range gate at which time instant to use
        if False:
            plt.subplot(221)
            plt.pcolormesh(n.arange(S.shape[0]),n.arange(S.shape[1]),10.0*n.log10(n.transpose(S)))
            plt.plot(n.arange(S.shape[0]),n.mod(best_rg0-best_drs,S.shape[1]))
            plt.subplot(222)
            line_mf=n.sum(best_s2,axis=0)
            noise_mf=n.median(line_mf)
            plt.plot(line_mf/noise_mf)
            plt.title("%d rg0 %1.2f km/s"%(best_rg0,best_rr))
            plt.subplot(223)
            
            plt.pcolormesh(10.0*n.log10(best_s2))
            plt.title(f)
            plt.subplot(224)
            plt.plot(ts,rrvec[max_dops]/1e3,".")
            plt.ylim([-10,100])
            plt.axhline(best_rr,color="red")
            plt.tight_layout()
            #plt.show()
            print("writing %s_slots.png"%(f))
            plt.savefig("%s_slots.png"%(f))
            plt.close()
            plt.clf()


    return


def analyze_range_dop_xc(z,o,rg,dop,codes):

    n_xspec=len(mc.pairs)
    S=n.zeros([z.shape[0],z.shape[1],n_xspec],dtype=n.complex64)
        
    for pi in range(len(mc.pairs)):
        pidx=mc.pairs[pi]
        for i in range(len(rg)):
            idx0=pidx[0]
            idx1=pidx[1]
#            cc0=n.fft.ifft(n.fft.fft(z[i,:,idx0],z.shape[1])*n.conj(n.fft.fft(codes[],z.shape[1])))
 #           S[2*i,:,pi]=cc0*n.conj(cc1)

            

def is_aliased(z,codes):
    """
    detect if range aliased or not.
    """
    # self correlations
    C=n.zeros([z.shape[0],z.shape[1],16,2],dtype=n.float64)
    for j in range(1):
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
    return(aliased)

        
    


def plot_3d(ls,ms,ts,rgs,ps,tm,prg,P,o,f):
    rgs=n.array(rgs)/1e3
    ls=n.array(ls)
    ps=n.array(ps)    
    ms=n.array(ms)
    ns=n.sqrt(1.0-ls**2.0-ms**2.0)
    
    if True:
        plt.subplot(221)
        plt.title(stuffr.unix2datestr(o["t0"]))
        plt.scatter(rgs*ls,rgs*ms,c=ts-n.min(ts))
#        plt.scatter(rgs*ls,rgs*ms,c=ps,vmin=15,vmax=50)#ts-n.min(ts))

        plt.xlim([-30,30])
        plt.ylim([-30,30])        
        plt.xlabel("East-West (km)")
        plt.ylabel("North-South (km)")        
        plt.subplot(222)
        plt.scatter(rgs*ls,rgs*ns,c=ts-n.min(ts))
#        plt.scatter(rgs*ls,rgs*ns,c=ps,vmin=15,vmax=50)#,c=ts-n.min(ts))        
        plt.xlim([-30,30])
#        plt.ylim([-5,5])        
        
        plt.xlabel("East-West (km)")
        plt.ylabel("Up (km)")        
        plt.subplot(223)
        plt.scatter(rgs*ms,rgs*ns,c=ts-n.min(ts))
        cb=plt.colorbar()
        cb.set_label("Time (s)")
        plt.xlabel("North-South (km)")
        plt.xlim([-30,30])        
        plt.ylabel("Up (km)")
        
        plt.subplot(224)
        plt.pcolormesh(tm,prg,P,vmin=0,vmax=50)
        cb=plt.colorbar()
        cb.set_label("Power/$\sigma$")
        plt.xlabel("Time (s)")
        plt.ylabel("Range (km)")
        plt.tight_layout()
        #        plt.show()
        print("writing %s.png"%(f))
        plt.savefig("%s.png"%(f))
        plt.clf()
        plt.close()
        ho=h5py.File("%s.h5"%(f),"w")
        ho["ts"]=ts
        ho["u"]=rgs*ls
        ho["v"]=rgs*ms
        ho["w"]=rgs*ns
        ho["power"]=ps
        ho.close()
        
#        plt.show()
        

        #plt.plot(ts,ns,".")
        


  #      del S
   #     del z
    #    del o
        #        plt.show()
                


#analyze_file("/data1/geminids/maarsy/GEMINIDS/20221212_023647112_event.ud3")
#analyze_file("/data1/geminids/maarsy/GEMINIDS/20221212_000027554_event.ud3")
#analyze_file("/data1/geminids/maarsy/GEMINIDS/20221212_000502142_event.ud3")
#analyze_file("/data1/geminids/maarsy/GEMINIDS/20221212_002707808_event.ud3")
#analyze_file("/data1/geminids/maarsy/GEMINIDS/20221212_003842936_event.ud3")
#analyze_file("/data1/geminids/maarsy/GEMINIDS/20221212_005801200_event.ud3")
#analyze_file("/data1/geminids/maarsy/GEMINIDS/20221212_011129196_event.ud3")
#analyze_file("/data1/geminids/maarsy/GEMINIDS/20221212_012915974_event.ud3")
#analyze_file("/data1/geminids/maarsy/GEMINIDS/20221212_021214752_event.ud3")
#analyze_file("/data1/geminids/maarsy/GEMINIDS/20221212_025010212_event.ud3")
#analyze_file("/data1/geminids/maarsy/GEMINIDS/20221212_032515792_event.ud3")
#analyze_file("/data1/geminids/maarsy/GEMINIDS/20221212_050541196_event.ud3")
#analyze_file("/data1/geminids/maarsy/GEMINIDS/20221212_052206360_event.ud3")
#analyze_file("/data1/geminids/maarsy/GEMINIDS/20221212_055125774_event.ud3")

#analyze_file("/data1/geminids/maarsy/GEMINIDS/20221212_010959126_event.ud3")
#slots_test("/data1/geminids/maarsy/GEMINIDS/20221212_010959126_event.ud3", snr_limit=25)
#exit(0)

if __name__ == "__main__":
    fl=glob.glob("/data1/geminids/maarsy/GEMINIDS/2*.ud3")
    fl.sort()
    for fi in range(rank, len(fl), size):
        f=fl[fi]
        print(f)
        slots_test(f)
        #analyze_file(f)
        #range_doppler_mf(f)



