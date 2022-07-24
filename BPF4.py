#coding:utf-8

#
# A class of IIR Band Pass Filter, process twice !
# (Target response is 2nd harmonic level less than -70dB)
#

import sys
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal

from iir1 import *
from ema1 import *

# Check version
#  Python 3.6.4 on win32 (Windows 10)
#  numpy 1.14.0 
#  matplotlib  2.1.1
#  scipy 1.4.1


class Class_BPFtwice(object):
    def __init__(self, fc=1000, gain=1.0, Q=40.0, sampling_rate=48000, moving_average_factor=None, down_sample_factor=None ):
        # initalize
        self.sr= sampling_rate
        self.fc= fc # center frequency of Band Pass Filter by unit is [Hz]
        self.gain= gain # magnification
        self.Q= Q   # Q factor
        # check Q
        if self.Q <= 0.0:
            print ('error: Q must be > 0.  filter becomes flat. (Class_BPF)')
            # sys.exit()
            self.a= np.array( [ 1.0, 0.0, 0.0])
            self.b= np.array( [ 1.0, 0.0, 0.0])
        else:
            self.a, self.b = self.bpf1()
        
        #-------------------------------------
        # set for filtering2
        #
        # Exponential Moving Average with Half-wave rectification, and smoothing via lpf
        if moving_average_factor is not None:
            self.maf= moving_average_factor
            self.ema= Class_EMA1(N=self.maf)
        else:
            self.ema= None
        # Down sampling to decrease temporal resolution
        if down_sample_factor is None:
            self.down_sample_factor= 1
        else:
            self.down_sample_factor= int(down_sample_factor)
        #
        #--------------------------------------
    
    def bpf1(self,):
        # primary digital filter
        a= np.zeros(3)
        b= np.zeros(3)
        
        wc= 2.0 * np.pi * self.fc / self.sr
        g0= 2.0 * np.tan( wc/2.0)
        
        a[0]=   4.0 +  2.0 * g0 / self.Q +  g0 * g0
        a[1]=  -8.0 + 2.0 * g0 * g0
        a[2]=   4.0 -  2.0 * g0 / self.Q +  g0 * g0
        
        b[0]=   2.0 * self.gain *  g0 / self.Q
        b[2]=  -2.0 * self.gain *  g0 / self.Q
        
        b /= a[0]
        a /= a[0]
        
        return  a,b
    
    def iir2(self,x):
        # calculate iir filter: x is input, y is output
        # y[0]= b[0] * x[0]  + b[1] * x[-1] + b[2] * x[-1]
        # y[0]= y[0] - a[1] * y[-1] - a[2] * y[-1]
        y= np.zeros(len(x))
        for n in range(len(x)):
            for i in range(len(self.b)):
                if n - i >= 0:
                    y[n] += self.b[i] * x[n - i]
            for j in range(1, len(self.a)):
                if n - j >= 0:
                    y[n] -= self.a[j] * y[n - j]
        return y
    
    def fone(self, xw):
        # calculate one point of frequecny response
        f= xw / self.sr
        yi= self.b[0] + self.b[1] * np.exp(-2j * np.pi * f) + self.b[2] * np.exp(-2j * np.pi * 2 * f)
        yb= self.a[0] + self.a[1] * np.exp(-2j * np.pi * f) + self.a[2] * np.exp(-2j * np.pi * 2 * f)
        val= yi/yb
        val= val * val
        return np.sqrt(val.real ** 2 + val.imag ** 2)
    
    def H0(self, freq_low=100, freq_high=7500, Band_num=256):
        # get Log scale frequecny response, from freq_low to freq_high, Band_num points
        amp=[]
        freq=[]
        bands= np.zeros(Band_num+1)
        fcl=freq_low * 1.0    # convert to float
        fch=freq_high * 1.0   # convert to float
        delta1=np.power(fch/fcl, 1.0 / (Band_num)) # Log Scale
        bands[0]=fcl
        #print ("i,band = 0", bands[0])
        for i in range(1, Band_num+1):
            bands[i]= bands[i-1] * delta1
            #print ("i,band =", i, bands[i]) 
        for f in bands:
            amp.append(self.fone(f))
        return   np.log10(amp) * 20, bands # = amp value, freq list
    
    def H0_show(self,freq_low=100, freq_high=7500, Band_num=256):
        # draw frequecny response
        plt.xlabel('Hz')
        plt.ylabel('dB')
        plt.title('Band Pass Filter')
        amp, freq=self.H0(freq_low=freq_low, freq_high=freq_high, Band_num=Band_num)
        plt.plot(freq, amp)
        plt.grid()
        plt.show()
    
    def filtering(self, xin):
        # filtering process, using scipy
        return signal.lfilter(self.b, self.a, self.filtering0(xin))
    
    def filtering0(self, xin):
        # filtering process, using scipy
        return signal.lfilter(self.b, self.a, xin)
    
    def filtering2(self, xin, dwn_len):
    	# xin should be mono
        # (1)filtering process, using scipy
        # (2)Exponential Moving Average with Half-wave rectification, and smoothing via lpf
        # (3)down sampling
        self.dwn_len=dwn_len
        return  np.resize( self.ema ( self.filtering(xin) ) , (self.dwn_len, self.down_sample_factor))[:,0]
    
    def check_minphase(self,):
        zeros, poles, _ = signal.tf2zpk(self.b, self.a)
        print ( zeros)
        print ( poles)
        for kai in np.concatenate([zeros,poles]):
            if not np.abs(kai) < 1.0:
                print ('This is not min phase')
    
    def f_show(self, worN=1024):
        # show frequency response, using scipy
        wlist, fres = signal.freqz(self.b, self.a, worN=worN)
        fres= fres * fres
        
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        flist = wlist / ((2.0 * np.pi) / self.sr)
        plt.title('frequency response')
        ax1 = fig.add_subplot(111)
        
        plt.semilogx(flist, 20 * np.log10(abs(fres)), 'b')  # plt.plot(flist, 20 * np.log10(abs(fres)), 'b')
        plt.ylabel('Amplitude [dB]', color='b')
        plt.xlabel('Frequency [Hz]')
        
        ax2 = ax1.twinx()
        angles = np.unwrap(np.angle(fres))
        angles = angles / ((2.0 * np.pi) / 360.0)
        plt.semilogx(flist, angles, 'g')  # plt.plot(flist, angles, 'g')
        plt.ylabel('Angle(deg)', color='g')
        plt.grid()
        plt.axis('tight')
        plt.show()
        
    def wav_show(self,y1,y2=None, y3=None):
    	# draw wavform
        plt.figure()
        plt.subplot(311)
        plt.xlabel('time step')
        plt.ylabel('amplitude')
        tlist= np.arange( len(y1) ) * (1 /self.sr)
        plt.plot( tlist, y1)
        
        if y2 is not None:
            plt.subplot(312)
            plt.xlabel('time step')
            plt.ylabel('amplitude')
            tlist= np.arange( len(y2) ) * (1 /self.sr)
            plt.plot( tlist, y2)
        
        if y3 is not None:
            plt.subplot(313)
            plt.xlabel('time step')
            plt.ylabel('amplitude')
            tlist= np.arange( len(y3) ) * (1 /self.sr)
            plt.plot( tlist, y3)
        
        plt.grid()
        plt.axis('tight')
        plt.show()

if __name__ == '__main__':
    
    from scipy import signal
    from scipy.io.wavfile import read as wavread
    # instance
    fc1=1000
    dsf=10
    Q0=40.0
    bpf=Class_BPFtwice(fc=fc1,  Q=Q0, sampling_rate=44100, moving_average_factor=80, down_sample_factor=dsf)
    
    
    # draw frequecny response
    bpf.H0_show(freq_high=20000)
    
    # draw frequecny response, using scipy
    bpf.f_show()
    
    # load a sample wav
    #path0='wav/400Hz-10dB_44100Hz_400msec.wav'
    #path0='wav/1KHz-10dB_44100Hz_400msec.wav'
    #path0='wav/3KHz-10dB_44100Hz_400msec.wav'
    #path0='wav/5KHz-10dB_44100Hz_400msec.wav'
    path0='wav/1KHz-10dB_44100Hz_400ms-TwoTube_stereo.wav'
    try:
        sr, y = wavread(path0)
    except:
        print ('error: wavread ')
        sys.exit()
    else:
        yg= y / (2 ** 15)
        if yg.ndim == 2:  # if stereo
            yg= np.average(yg, axis=1)
        print ('sampling rate ', sr)
        print ('y.shape', yg.shape)
    
    y2=bpf.filtering( yg)   # iir2( yg)    
    
    # Exponential Moving Average with Half-wave rectification
    ema1= Class_EMA1()
    y3=ema1( y2)
    bpf.wav_show(yg, y2, y3)
    
    # compare both for check
    y5=bpf.filtering2(yg, int(len(yg)/dsf))
    print ('y5.shape', y5.shape)
    bpf.wav_show(yg,y3,y5)
    
    

