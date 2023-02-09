import scipy.constants as c
import numpy as n
import re
import matplotlib.pyplot as plt

coords={"lat":69.29836217360676,
        "lon":16.04139069818655,
        "alt":1.0}

ipp=1e-3
txlen=16

def get_codes():
    """
    16-bit complementary code pair used at MAARSY
    """
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
        z2[:,:,i]=z2[:,:,i]*n.exp(1j*phases_rad[i])
        
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


def is_aliased(z,codes):
    """
    Detect if echo is range aliased or not.
    Use the full antenna channel (0) only
    """
    # self correlations
    C=n.zeros([z.shape[0],z.shape[1],2],dtype=n.float64)
    for i in range(int(z.shape[0]/2)):
        # not aliased. 
        cc0=n.fft.ifft(n.fft.fft(z[2*i,:,0],z.shape[1])*n.conj(n.fft.fft(codes[0],z.shape[1])))
        C[2*i,:,0]=n.real(cc0*n.conj(cc0))
        cc0=n.fft.ifft(n.fft.fft(z[2*i+1,:,0],z.shape[1])*n.conj(n.fft.fft(codes[1],z.shape[1])))
        C[2*i+1,:,0]=n.real(cc0*n.conj(cc0))
        
        # range aliased
        cc0=n.fft.ifft(n.fft.fft(z[2*i,:,0],z.shape[1])*n.conj(n.fft.fft(codes[1],z.shape[1])))
        C[2*i,:,1]=n.real(cc0*n.conj(cc0))
        cc0=n.fft.ifft(n.fft.fft(z[2*i+1,:,0],z.shape[1])*n.conj(n.fft.fft(codes[0],z.shape[1])))
        C[2*i+1,:,1]=n.real(cc0*n.conj(cc0))

    # extract range gate with maximum power for each time step
    # this leaves pwr0 and pwr1 as power as a function of time
    pwr0=n.max(C[:,:,0],axis=1)
    pwr1=n.max(C[:,:,1],axis=1)  

    aliased=False
    if n.sum(pwr1) > n.sum(pwr0):
        aliased=True
    return(aliased)


    

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

fradar=53.5e6 # center frequency in Hz
sampling_rate=0.5e6
range_gate = c.c/sampling_rate/2.0 # range-step in meters
lam=c.c/fradar # wavelength in meters
phases_deg=n.array([160.706296, # 433
                    17.715985,  # A
                    5.499101,   # B
                    0,          # C
                    -5.865708,  # D
                    -12.842389, # E
                    4.229617,   # F
                    -8.033325,  # M
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

# high res
pairs=[(7,1),(4,7),(6,5),(2,3),
       (7,2),(6,1),(5,7),(4,3),
       (7,3),(1,2),(6,7),(5,4)]


pairs_lr=[(9,15),(9,10),(10,15)]
weights_lr=[1,1,1]

other_weight=4.0
weights=[other_weight/4, other_weight/4, other_weight/4.0, other_weight/4, # 2-8 alignment
         other_weight/4, other_weight/4, other_weight/4.0, other_weight/4, # 8-3 alignment
         other_weight/4, other_weight/4, other_weight/4.0, other_weight/4] # 8-4 alignment         


def get_beam_pattern_hexagon():
    return(get_beam_pattern(fname="cal/maarsy_antenna_02.txt"))
def get_beam_pattern_full():
    return(get_beam_pattern(fname="cal/maarsy_antenna_00.txt"))
def get_beam_pattern_anemone():
    return(get_beam_pattern(fname="cal/maarsy_antenna_01.txt"))

def get_beam_pattern(fname="cal/maarsy_antenna_02.txt"):
    f=open(fname,"r")
    G=n.zeros([201,201])
    ri=0
    u=n.zeros(201)
    for l in f.readlines():
#        print(re.split('[\t ]+',l.strip()))
        arr=re.split('[\t ]+',l.strip())
        if arr[0] == 'Dcosx/Dcosy':
            u=n.array(arr[1:len(arr)],dtype=n.float32)
        elif len(arr) == 202 and arr[0] != 'Dcosx/Dcosy':
            row=n.array(arr,dtype=n.float32)
            G[ri,:]=row[1:len(row)]
            ri+=1
    G[G<1e-3]=1e-4
    f.close()
    angles=180.0*(n.pi/2.0-n.arccos(u))/n.pi
    plt.pcolormesh(u,u,10.0*n.log10(G))
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    plot_beam_patterns()

def plot_beam_patterns():    
    # null at 0.6, unambiguous up to 0.3
    get_beam_pattern_hexagon()
    # null at 0.23 unambiguous up to 0.1
    get_beam_pattern_anemone()
    # null at 0.08
    get_beam_pattern_full()
    
