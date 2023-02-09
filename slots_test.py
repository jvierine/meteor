#!/usr/bin/env python3
import matplotlib
#matplotlib.use('Agg')
import numpy as n
import matplotlib.pyplot as plt
import glob
import scipy.interpolate as sint
import itertools
from mpi4py import MPI
import scipy.constants as c
import stuffr
import h5py
import scipy.optimize as sio

import maarsy_config as mc
import maarsy_interferometry as mi

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()





def median_std(S):
    mean_est=n.nanmedian(S)
    std_est=1.48*n.nanmedian(n.abs(S-mean_est))
    return(std_est)



def estimate_pp_doppler(z, max_dops, best_drs, mask,idx, max_ti, best_rr, best_rgs, incoh_dec=4):
    # z is voltage
    # mask is mf mask
    # idx is sampling index for range-Doppler array
    # max_rgs is the range-gates where echoes are observed
    # txlen is length of tx pulse
    # fftfactor is how much zero padding to do
    # max_dops is which doppler index of fft
    # incoh_dec is how much to incoherently average
    # max_rgs is the range-gates with maximum power
    # range shift as a functino of time

    fftfactor=32
    codes=mc.get_codes()
    txlen=mc.txlen
    nrg=z.shape[1]
    
    D=n.zeros([z.shape[0],z.shape[1]],dtype=n.complex64)            

    # Estimate pulse-to-pulse Doppler

    for i in range(len(max_dops)-1): # pulse-to-pulse subtracts one time step
        for k in range(incoh_dec):
            F0=n.fft.fft(mask*z[i+k,idx,0]*n.conj(codes[(i+k)%2]),txlen*fftfactor,axis=1)
            F1=n.fft.fft(mask*z[i+k+1,idx,0]*n.conj(codes[(i+k+1)%2]),txlen*fftfactor,axis=1)
                    
            fxc = (F0*n.conj(F1))[:,max_dops[i]]
                    
            # take the max doppler bin of ith time step
            # roll to allow averaging
            D[i,:] += n.roll(fxc,best_drs[i])

    for i in range(len(max_dops)-1): # pulse-to-pulse subtracts one time step
        D[i,:] = n.roll(D[i,:],-best_drs[i])

    zpp = []
    for i in range(len(max_dops)-1): # pulse-to-pulse subtracts one time step
        if (best_rgs[i] > 0) and (best_rgs[i] < (nrg-1)):
            vec = D[i,(best_rgs[i]-1):(best_rgs[i]+1)]
            val=vec[n.argmax(n.abs(vec))]
        else:
            val=D[i,best_rgs[i]]
        zpp.append(val)

    # best one to use
    dphi=n.angle(zpp)

    # look for phase chirp with polynomial chirp-rate
    # ddphi negative
    #dphi = dphi0 + ddphi*(t-t0) + dddphi*(t-t0)**2.0 + ddddphi*(t-t0)**3.0

    # at maximum snr, we should have approximately this dphi0
    dphi0_est = dphi[max_ti]

    # estimate change in phase using snr weighted mean of differences
    weights = n.abs(zpp[0:(len(zpp)-1)])+n.abs(zpp[1:(len(zpp))])
    ddphi0_est=n.angle(n.sum(weights*n.exp(1j*dphi[1:(len(dphi))])*n.conj(n.exp(1j*dphi[0:(len(dphi)-1)])))/n.sum(weights))/mc.ipp

    # time vector with 0 shifted t- peak snr time
    ts = n.arange(len(dphi))*mc.ipp
    tm = ts-ts[max_ti]

    # model
    def model(x):
        dphi0=x[0]
        ddphi=x[1]
        dddphi=x[2]
        ddddphi=x[3]        
        return( n.exp(1j*(dphi0 + ddphi*tm + dddphi*tm**2.0 + ddddphi*tm**3.0)))

    # snr weighted 2pi wrapping least-squares
    weights=n.abs(zpp)
    def ss(x):
        return(n.sum(weights*n.abs(n.angle(model(x)*n.conj(n.exp(1j*dphi))))**2.0))

    # fmin search
    xhat=sio.fmin(ss,[dphi0_est,ddphi0_est,0,0])
    xhat=sio.fmin(ss,xhat)

    # debug plot of fitting pulse-to-pulse doppler
    if False:
        plt.plot(n.angle(model(xhat)),".")
        plt.title(xhat)
        plt.plot(dphi,".")
        plt.show()
        plt.plot(n.unwrap(n.angle(model(xhat))),".")
        plt.title(xhat)
        plt.show()

    dphi_model = n.unwrap(n.angle(model(xhat)))
    # radial velocity from pulse-to-pulse doppler
    vpp=-(dphi_model/mc.ipp)*mc.lam/4.0/n.pi

    vpp_meas=-(dphi/mc.ipp)*mc.lam/4.0/n.pi    

    # try these aliases
    plot_rr_alias=True
    n_aliases=70
    AV = n.zeros([len(zpp),n_aliases])

    best_alias = 0
    best_distance=1e99
    for alias_idx in range(n_aliases):
        # start with negative values to allow slightly negative range-rates
        ai=alias_idx - 10
        # 2.0*f*v/c = df
        # v = c.c*1e3*alias_idx/2.0/mc.fradar
        alias_vels=(vpp + c.c*(1.0/mc.ipp)*ai/2.0/mc.fradar)/1e3
        alias_vels_m=(vpp_meas + c.c*(1.0/mc.ipp)*ai/2.0/mc.fradar)/1e3        
        AV[:,alias_idx]=alias_vels

        # todo: snr weighted vel should be closest to best_rr
        weights=n.abs(zpp)
        model_vel = n.sum(weights*alias_vels)/n.sum(weights) 
        if n.abs(model_vel - best_rr) < best_distance:
            best_distance=n.abs(model_vel - best_rr)#n.abs(alias_vels[max_ti] - best_rr)
            best_alias=ai
        
        if plot_rr_alias:
#            if n.max(n.abs(alias_vels[max_ti]-best_rr)) < 10:
 #               plt.plot(n.arange(len(alias_vels))/1e3,alias_vels,label="%d"%(alias_idx))
            if n.max(n.abs(alias_vels_m[max_ti]-best_rr)) < 10:                
                plt.plot(n.arange(len(alias_vels))/1e3,alias_vels_m,".",label="%d"%(alias_idx))

                # add aliases

    if plot_rr_alias:
        plt.axvline(max_ti*mc.ipp)

        alias_vels=(vpp + c.c*(1.0/mc.ipp)*best_alias/2.0/mc.fradar)/1e3
        plt.plot(n.arange(len(alias_vels))/1e3,alias_vels,label="best",color="black")
        
        plt.title("Course range-rate estimate %1.2f km/s\ndist %1.2f km/s"%(best_rr,best_distance))
        plt.axhline(best_rr)
        plt.xlabel("Time (s)")
        plt.ylabel("Velocity (km/s)")
        plt.legend()
        plt.show()

    # find the 
            
    return(D)



def initial_range_dop_mf(z,incoh_dec=4):
    # for each range
    codes=mc.get_codes()
    sr=mc.sampling_rate
    txlen=mc.txlen
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

#    Phi=n.zeros([z.shape[0],z.shape[1]],dtype=n.complex64)                
    freqvec=n.fft.fftfreq(fftfactor*txlen,d=1.0/sr)
    # 2*fr*v/c=df
    # => df*c/fr/2 = v
    rrvec=c.c*n.fft.fftfreq(fftfactor*txlen,d=1.0/sr)/mc.fradar/2.0
    for i in range(int(z.shape[0]-incoh_dec+1)):
        MF[:,:]=0.0
        for k in range(incoh_dec): # incoh integration. use very little initially!
            for j in range(1):
                MF+=n.abs(n.fft.fft(mask*z[i+k,idx,j]*n.conj(codes[(i+k)%2]),txlen*fftfactor,axis=1))**2.0
        max_rg,max_dop=n.unravel_index(n.argmax(MF),MF.shape)
        S[i,:]=n.max(MF,axis=1)
        D[i,:]=rrvec[n.argmax(MF,axis=1)]
        max_rgs.append(max_rg)
        max_dops.append(max_dop)
        max_ss.append(S[i,max_rg])
        ts.append(i)

    return(S, max_rgs, max_dops, max_ss, mask, idx)

def coarse_radial_distance(S,
                           rrmax=200.0, # maximum allowed range-rate (km/s)
                           rrmin=-20):  # minimum allowed range-rate (km/s)
    """
    Figure out SNR weighted average range-rate from range-migration of echo
    this will later be used to guide selection of echoes to estimate pulse-to-pulse
    doppler shift, and to disambiguate it.
    """
    tmax=S.shape[0]*mc.ipp

    # range-rate step
    # make it so that the range migration between two different
    # tries is at most 150 meters, which is half of a range-gate
    drr = 0.15/tmax
    n_rr=int((rrmax-rrmin)/drr)
    rrtest = n.linspace(rrmin,rrmax,num=n_rr)
    tvec=n.arange(S.shape[0])*mc.ipp
    best=0

#    std_est=median_std(S)
    
    for i in range(len(rrtest)):
        # grid search all range-rates

        # all possible range-rates in units of
        
        # range as range-gates for each inter-pulse period
        drs=n.array(n.round(1e3*rrtest[i]*tvec/mc.range_gate),dtype=int)
        
        S2=n.copy(S)
        for ti in range(S.shape[0]):
            S2[ti,:]=n.roll(S2[ti,:],drs[ti])
            
        mf=n.max(n.sum(S2,axis=0))
        if mf > best:
            best=mf
            best_rr=rrtest[i]
            best_s2=S2
            best_drs=drs
            best_rg0=n.argmax(n.sum(best_s2,axis=0))

    best_rgs = n.zeros(S.shape[0],dtype=int)
    for i in range(S.shape[0]):
        best_rgs[i]=n.mod(best_rg0-best_drs[i],S.shape[1])

    # this is the time index corresponding to maximum SNR
    # we will use this to disambiguate pulse-to-pulse doppler

    # snr weighted mean t0
    max_ti=int(n.round(n.sum(best_s2[:,best_rg0]*n.arange(S.shape[0]))/n.sum(best_s2[:,best_rg0])))
#    max_ti=n.argmax(best_s2[:,best_rg0])
        
    if False:
        plt.title("%1.2f km/s"%(best_rr))
        plt.pcolormesh(n.transpose(10.0*n.log10(S)))
        plt.plot(best_rgs,".")    
        plt.show()
        plt.plot(best_s2[:,best_rg0],label="0")
        plt.plot(best_s2[:,best_rg0+1],label="+1")
        plt.plot(best_s2[:,best_rg0-1],label="-1")
        
        plt.legend()
        plt.show()

    # estimate peak signal-to-noise ratio
    noise_floor=n.median(best_s2)
    peak_snr = (best_s2[max_ti,best_rg0]-noise_floor)/noise_floor

    
        
    return(best_rr, best_rgs, best_drs, max_ti, peak_snr, noise_floor)
        

def slots_test(f, snr_limit=25, sr=0.5e6, incoh_dec=16, N=200):
    
    codes=mc.get_codes()
    z,o=mc.read_ud3(fname=f)
    
    aliased=mc.is_aliased(z,codes)
    
    if aliased:
        print("range aliased echo detected. ignoring")
        return


    S,max_rgs,max_dops,max_ss,mask,idx=initial_range_dop_mf(z,incoh_dec=4)

    # get an estimate the the range-rate from range-gate migration
    best_rr, best_rgs, best_drs, max_ti, peak_snr, noise_floor = coarse_radial_distance(S)


    plt.pcolormesh(n.transpose(10.0*n.log10(n.abs(S))))
    plt.colorbar()
    plt.show()
    
    print("peak range-gate %d range-rate %1.2f km/s peak snr %1.2f"%(best_rgs[max_ti],best_rr,peak_snr))

    D = estimate_pp_doppler(z, max_dops, best_drs, mask, idx, max_ti, best_rr, best_rgs, incoh_dec=4)
    
    return
    
    if n.abs(best_rr) > 4.0 and peak_snr > 1.5:
        
        # only analyze if there is enough doppler velocity
        # unaliased doppler range
        
        if True:


            # interferometry setup
            x,y=mc.antenna_pos()
            pairs=mc.pairs_lr
            weights=mc.weights_lr
            
            # low res
            u,v,ui,vi=mi.uv_coverage(x,y,pairs,N=N)
            # 0.3 is the maximum unambiguous angle for hexagons
            kvec_x,kvec_y,lv,mv,direction_mask=mi.kvecs(k=2.0*n.pi/mc.lam,N=N,maxdcos=0.3)
            
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
            
            n_xspec=len(pairs)
            XC=n.zeros([z.shape[0],z.shape[1],n_xspec],dtype=n.complex64)

            # figure out the range gate
            aliased_rg0 = n.mod(best_rg0,S.shape[1])

            

            # calculate cross-correlations
            for i in range(len(max_rgs)):
                for pi in range(len(pairs)):
                    pidx=pairs[pi]
                    idx0=pidx[0]
                    idx1=pidx[1]
                    for k in range(incoh_dec):
                        F0=n.fft.fft(mask*z[i+k,idx,idx0]*n.conj(codes[(i+k)%2]),txlen*fftfactor,axis=1)
                        F1=n.fft.fft(mask*z[i+k,idx,idx1]*n.conj(codes[(i+k)%2]),txlen*fftfactor,axis=1)
                        
                        fxc = (F0*n.conj(F1))[:,max_dops[i]]

                        # take the max doppler bin of ith time step
                        XC[i,:,pi] += n.roll(fxc,best_drs[i])


            # do we remove the background phase.
            # I am unsure if this should or should not be removed.
            if True:
                for pi in range(len(pairs)):
                    XC[:,:,pi]=XC[:,:,pi]-n.median(XC[:,:,pi])


            P=n.sum(n.abs(XC[:,:,:]),axis=2)
#            plt.pcolormesh(10.0*n.log10(P))
 #           plt.show()
                
            best_rgs=[]
            rg_idx=[aliased_rg0-1,aliased_rg0,aliased_rg0+1]
            for i in range(len(max_rgs)):
                best_rgs.append(rg_idx[n.argmax([P[i,aliased_rg0-1],P[i,aliased_rg0],P[i,aliased_rg0+1]])])
                
            for pi in range(len(pairs)):
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
            for pi in range(len(pairs)):
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
                    l,m=mi.find_angle(u,v,XC[i,best_rgs[i],:],kvec_x,kvec_y,lv,mv,weights=weights,mask=direction_mask)
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

    n_xspec=len(pairs)
    S=n.zeros([z.shape[0],z.shape[1],n_xspec],dtype=n.complex64)
        
    for pi in range(len(mc.pairs)):
        pidx=mc.pairs[pi]
        for i in range(len(rg)):
            idx0=pidx[0]
            idx1=pidx[1]
#            cc0=n.fft.ifft(n.fft.fft(z[i,:,idx0],z.shape[1])*n.conj(n.fft.fft(codes[],z.shape[1])))
 #           S[2*i,:,pi]=cc0*n.conj(cc1)

            


        
    


def plot_3d(ls,ms,ts,rgs,ps,tm,prg,P,o,f):
    rgs=n.array(rgs)/1e3
    ls=n.array(ls)
    ps=n.array(ps)    
    ms=n.array(ms)
    ns=n.sqrt(1.0-ls**2.0-ms**2.0)
    
    if True:
        if len(ls) < 5:
            return
        plt.subplot(221)
        plt.title(stuffr.unix2datestr(o["t0"]))
        plt.scatter(rgs*ls,rgs*ms,c=ts-n.min(ts))
#        plt.scatter(rgs*ls,rgs*ms,c=ps,vmin=15,vmax=50)#ts-n.min(ts))

    #    plt.xlim([-30,30])
     #   plt.ylim([-30,30])        
        plt.xlabel("East-West (km)")
        plt.ylabel("North-South (km)")        
        plt.subplot(222)
        plt.scatter(rgs*ls,rgs*ns,c=ts-n.min(ts))
#        plt.scatter(rgs*ls,rgs*ns,c=ps,vmin=15,vmax=50)#,c=ts-n.min(ts))        
  #      plt.xlim([-30,30])
   #     plt.ylim([70,130])        
        
        plt.xlabel("East-West (km)")
        plt.ylabel("Up (km)")
        
        plt.subplot(223)
        plt.scatter(rgs*ms,rgs*ns,c=ts-n.min(ts))
        cb=plt.colorbar()
        cb.set_label("Time (s)")
        plt.xlabel("North-South (km)")
 #       plt.xlim([-30,30])
#        plt.ylim([70,130])        
        plt.ylabel("Up (km)")
        
        plt.subplot(224)
        plt.pcolormesh(tm,prg,P,vmin=0,vmax=50)
        cb=plt.colorbar()
        cb.set_label("Power/$\sigma$")
        plt.xlabel("Time (s)")
        plt.ylabel("Range (km)")
        plt.tight_layout()
        #        plt.show()
        print("writing %s_slots.png"%(f))
        plt.savefig("%s_slots.png"%(f))
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
                

if True:
    slots_test("/data1/geminids/maarsy/GEMINIDS/20221212_023647112_event.ud3")
    slots_test("/data1/geminids/maarsy/GEMINIDS/20221212_000027554_event.ud3")
    slots_test("/data1/geminids/maarsy/GEMINIDS/20221212_000502142_event.ud3")
    slots_test("/data1/geminids/maarsy/GEMINIDS/20221212_002707808_event.ud3")
    slots_test("/data1/geminids/maarsy/GEMINIDS/20221212_003842936_event.ud3")
    slots_test("/data1/geminids/maarsy/GEMINIDS/20221212_005801200_event.ud3")
    slots_test("/data1/geminids/maarsy/GEMINIDS/20221212_011129196_event.ud3")
    slots_test("/data1/geminids/maarsy/GEMINIDS/20221212_012915974_event.ud3")
    slots_test("/data1/geminids/maarsy/GEMINIDS/20221212_021214752_event.ud3")
    slots_test("/data1/geminids/maarsy/GEMINIDS/20221212_025010212_event.ud3")
    slots_test("/data1/geminids/maarsy/GEMINIDS/20221212_032515792_event.ud3")
    slots_test("/data1/geminids/maarsy/GEMINIDS/20221212_050541196_event.ud3")
    slots_test("/data1/geminids/maarsy/GEMINIDS/20221212_052206360_event.ud3")
    slots_test("/data1/geminids/maarsy/GEMINIDS/20221212_055125774_event.ud3")
    
    #analyze_file("/data1/geminids/maarsy/GEMINIDS/20221212_010959126_event.ud3")
    #slots_test("/data1/geminids/maarsy/GEMINIDS/20221212_010959126_event.ud3", snr_limit=25)
    slots_test("/data1/geminids/maarsy/GEMINIDS/20221212_055125774_event.ud3", snr_limit=25)
    exit(0)

if __name__ == "__main__":
    fl=glob.glob("/data1/geminids/maarsy/GEMINIDS/2*.ud3")
    fl.sort()
    for fi in range(rank, len(fl), size):
        f=fl[fi]
        print(f)
        slots_test(f)
        #analyze_file(f)
        #range_doppler_mf(f)



