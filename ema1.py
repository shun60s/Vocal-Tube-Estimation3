#coding:utf-8

# A class of Exponential Moving Average with Half-wave rectification, and smoothing via lpf
#
# Half-wave rectification until a few KHz signal.
# More than a few KHz signal is transformed to DC with ripple signal. And smooth ripple signal.


# Check version
#  Python 3.6.4 on win32 (Windows 10)
#  numpy 1.16.3
#  matplotlib  2.1.1
#  scipy 1.4.1

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import lfilter
from iir1 import *

class Class_EMA1(object):
    def __init__(self, N=80, Clip_bottom=0.0, PrintOut=False):
        # initalize
        self.N= N
        self.alfa= 2.0 / (self.N + 1.0)
        self.Clip_bottom= Clip_bottom
        
        if PrintOut:
            print ('alfa (EMA)', self.alfa)
            print ('half cycle (EMA)', self.N/2.8854)
        
    def __call__(self, x, sr=48000, smooth=True):
        # x dimension should be 1-zigenn
        # Half-wave rectification
        y= np.clip(x, self.Clip_bottom, None)
        
        
        '''
        # output numpy array
        y2= np.empty( (len(x)), dtype=np.float32)
        y2[0]= self.alfa * y[0]
        for i in range( len(x) -1 ):
            y2[i+1] = self.alfa *  y[i] + (1.0- self.alfa) * y2[i]
        '''
        
        #  use scipy's lfilter([b]          [a]
        y2,_ = lfilter( [self.alfa, 0.0],[1.0,self.alfa - 1.0], y, zi=[ y[0] * (1.0- self.alfa)])
        
        if smooth:
            return self.smoothing(y2, sr)
        else:
            return y2
        
    def smoothing(self,x, sr=48000):
    	# smoothing via lpf
        self.lpf=Class_IIR1(fc= sr / 30 , n_order=3, sampling_rate=sr)
        return self.lpf.filtering(x)
        
    def wav_show(self,y1,y2=None, y3=None, sr=48000):
    	# draw wavform
        plt.figure()
        plt.subplot(311)
        plt.xlabel('time step')
        plt.ylabel('amplitude')
        tlist= np.arange( len(y1) ) * (1 / sr)
        plt.plot( tlist, y1)
        
        if y2 is not None:
            plt.subplot(312)
            plt.xlabel('time step')
            plt.ylabel('amplitude')
            tlist= np.arange( len(y2) ) * (1 /sr)
            plt.plot( tlist, y2)
        
        if y3 is not None:
            plt.subplot(313)
            plt.xlabel('time step')
            plt.ylabel('amplitude')
            tlist= np.arange( len(y3) ) * (1 / sr)
            plt.plot( tlist, y3)
        
        plt.grid()
        plt.axis('tight')
        plt.show()

if __name__ == '__main__':
    #
    from scipy import signal
    from scipy.io.wavfile import read as wavread
    # instance
    ema1= Class_EMA1(PrintOut=True)
    
    # load a sample wav
    #path0='wav/400Hz-10dB_44100Hz_400msec.wav'
    #path0='wav/1KHz-10dB_44100Hz_400msec.wav'
    path0='wav/3KHz-10dB_44100Hz_400msec.wav'
    #path0='wav/5KHz-10dB_44100Hz_400msec.wav'
    try:
        sr, y = wavread(path0)
    except:
        print ('error: wavread ')
        sys.exit()
    else:
        yg= y / (2 ** 15)
        print ('sampling rate ', sr)
        print ('y.shape', yg.shape)
    
    # process
    y2=ema1( yg, sr, smooth=False)
    y3=ema1.smoothing( y2, sr)
    
    # draw wav
    ema1.wav_show(yg, y2, y3, sr=sr)



