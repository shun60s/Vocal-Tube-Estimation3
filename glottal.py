#coding:utf-8

# glottal voice source as input of Two Tubes Model of vocal tract
# Glottal Volume Velocity 
# based on A.E.Rosenberg's formula as Glottal Volume Velocity

# introduce repeated glottal voice source to explain harmonic structure in the frequency spectrum.

import numpy as np
from matplotlib import pyplot as plt

# Check version
#  Python 3.10.4 on win32 (Windows 10)
#  numpy 1.21.6 
#  matplotlib  3.5.2


class Class_Glottal(object):
    def __init__(self, tclosed=5.0, trise=6.0, tfall=2.0, sampling_rate=48000, F0=None):
        # initalize
        self.sr= sampling_rate
        self.F0= F0
        if self.F0 is not None:  # if F0 frequency is specified.
            ratio= (1/F0) / ((tclosed + trise + tfall) /1000) 
            self.tclosed=tclosed * ratio  # duration time of close state [mSec]
            self.trise=trise * ratio      # duration time of opening [mSec]
            self.tfall=tfall * ratio      # duration time of closing [mSec]
        else:
            self.tclosed=tclosed  # duration time of close state [mSec]
            self.trise=trise      # duration time of opening [mSec]
            self.tfall=tfall      # duration time of closing [mSec]
        
        self.yg=self.make_one_plus()
        
    def make_one_plus(self,):
        # output yg
        self.N1=int( (self.tclosed / 1000.) * self.sr )
        self.N2=int( (self.trise / 1000.) * self.sr )
        self.N3=int( (self.tfall / 1000.) * self.sr )
        self.LL= self.N1+ self.N2 + self.N3
        if self.F0 is not None:
            print ('digitized F0 is', self.sr / self.LL)
        yg=np.zeros(self.LL)
        #print ('Length= ', self.LL)
        for t0 in range(self.LL):
            if t0 < self.N1 :
                pass
            elif t0 <= (self.N2 + self.N1):
                yg[t0]= 0.5 * ( 1.0 - np.cos( ( np.pi / self.N2 ) * (t0 - self.N1)) )
            else:
                yg[t0]= np.cos( ( np.pi / ( 2.0 * self.N3 )) * ( t0 - (self.N2 + self.N1) )  )
        return yg

    def make_N_repeat(self, repeat_num=3):
        yg_repeat=np.zeros( len(self.yg) * repeat_num)
        for loop in range( repeat_num):
            yg_repeat[len(self.yg)*loop:len(self.yg)*(loop+1)]= self.yg
        return  yg_repeat
    
    def fone(self, f):
        # calculate one point of frequecny response
        xw= 2.0 * np.pi * f / self.sr
        yi=0.0
        yb=0.0
        for v in range (0,(self.N2 + self.N3)):
            yi+=  self.yg[self.N1 + v] * np.exp(-1j * xw * v)
            yb+=  self.yg[self.N1 + v]
        val= yi/yb
        return np.sqrt(val.real ** 2 + val.imag ** 2)
    
    def H0(self, freq_low=100, freq_high=5000, Band_num=256, freq_list=None):
        # get Log scale frequecny response, from freq_low to freq_high, Band_num points
        #
        if freq_list is not None:
            bands= freq_list
        else:
            bands= np.zeros(Band_num+1)
            fcl=freq_low * 1.0    # convert to float
            fch=freq_high * 1.0   # convert to float
            delta1=np.power(fch/fcl, 1.0 / (Band_num)) # Log Scale
            bands[0]=fcl
            #print ("i,band = 0", bands[0])
            for i in range(1, Band_num+1):
                bands[i]= bands[i-1] * delta1
                #print ("i,band =", i, bands[i]) 
            
        amp=self.fone(bands)
        return   np.log10(amp) * 20, bands # = amp value, freq list
    
    #
    # introduce repeated glottal voice sourcel to explain harmonic structure in the frequency spectrum.
    #
    def fone_N_repeat(self, f, N_repeat):
        # calculate one point of frequecny response of N repeated signal
        xw= 2.0 * np.pi * f / self.sr
        yi=0.0
        yb=0.0
        for v in range (0,self.LL):
            for i in range (0, N_repeat):
                yi+=  self.yg[v] * np.exp(-1j * xw  * (self.LL * i + v))
                yb+=  self.yg[v]
        val= yi/yb
        return np.sqrt(val.real ** 2 + val.imag ** 2)
    
    def H0_N_repeat(self, N_repeat=3, freq_low=100, freq_high=5000, Band_num=256, freq_list=None):
        # get Log scale frequecny response, from freq_low to freq_high, Band_num points
        self.N_repeat=N_repeat
        #
        if freq_list is not None:
            bands= freq_list
        else:
            bands= np.zeros(Band_num+1)
            fcl=freq_low * 1.0    # convert to float
            fch=freq_high * 1.0   # convert to float
            delta1=np.power(fch/fcl, 1.0 / (Band_num)) # Log Scale
            bands[0]=fcl
            #print ("i,band = 0", bands[0])
            for i in range(1, Band_num+1):
                bands[i]= bands[i-1] * delta1
                #print ("i,band =", i, bands[i]) 
            
        amp= self.fone_N_repeat(bands, self.N_repeat)
        return   np.log10(amp) * 20, bands # = amp value, freq list

if __name__ == '__main__':
    
    # instance
    #glo=Class_Glottal()
    glo=Class_Glottal(F0=115, sampling_rate=48000*4)  # for precision, higher sampling_rate is better
    
    # draw
    fig = plt.figure()
    # draw one waveform
    plt.subplot(3,1,1)
    plt.xlabel('mSec')
    plt.ylabel('level')
    plt.title('Glottal Waveform')
    plt.plot( (np.arange(len(glo.yg)) * 1000.0 / glo.sr) , glo.yg)

    # draw frequecny response
    plt.subplot(3,1,2)
    plt.xlabel('Hz')
    plt.ylabel('dB')
    plt.title('Glottal frequecny response')
    amp, freq=glo.H0(freq_high=5000, Band_num=1024)
    amp_repeat, freq_repeat=glo.H0_N_repeat(N_repeat=5, freq_high=5000, Band_num=1024)
    plt.plot(freq, amp, label='one pulse')
    # show harmonic structure in the frequency spectrum
    plt.plot(freq_repeat, amp_repeat, color='r', label=str(glo.N_repeat) + '-repeat')
    plt.legend()
    plt.grid()
    
    # draw repeated waveform
    yg_repeat=glo.make_N_repeat(repeat_num=3)
    plt.subplot(3,1,3)
    plt.xlabel('mSec')
    plt.ylabel('level')
    plt.title('Glottal repeated Waveform')
    plt.plot( (np.arange(len(yg_repeat)) * 1000.0 / glo.sr) , yg_repeat)
    
    #
    fig.tight_layout()
    plt.show()
    
