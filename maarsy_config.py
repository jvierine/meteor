import scipy.constants as c
import numpy as n
coords={"lat":69.29836217360676,
        "lon":16.04139069818655,
        "alt":1.0}

ipp=1e-3


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
lam=c.c/53.5e6
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

pairs=[(9,15),(9,10),(10,15),
       (7,1),(4,7),(6,5),(2,3),
       (7,2),(6,1),(5,7),(4,3),
       (7,3),(1,2),(6,7),(5,4)]

other_weight=0.25
weights=[1,1,1,
         other_weight/4, other_weight/4, other_weight/4.0, other_weight/4, # 2-8 alignment
         other_weight/4, other_weight/4, other_weight/4.0, other_weight/4, # 8-3 alignment
         other_weight/4, other_weight/4, other_weight/4.0, other_weight/4] # 8-4 alignment         

