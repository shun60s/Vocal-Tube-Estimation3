#coding:utf-8

# A class of mel-frequency equal shifted list

# Check version
#  Python 3.6.4 on win32 (Windows 10)
#  numpy 1.16.3
#  matplotlib  2.1.1


import numpy as np


class Class_mel(object):
    def __init__(self, num_band=1024, fmin=40, fmax=8000):
        # initalize
        self.num_band=num_band # number of equal shift band
        self.fmin= fmin # low frequecny limit by unit is [Hz]
        self.fmax= fmax # low frequecny limit by unit is [Hz]
        self.mel_fmin= self.hz2mel( self.fmin ) # low  side center frequecny by unit is [mel frequency]
        self.mel_fmax= self.hz2mel( self.fmax ) # high side center frequecny by unit is [mel frequency]
        self.mel_flist= np.linspace( self.mel_fmin, self.mel_fmax, self.num_band)
        #print ( self.mel_flist)
        self.flist = self.mel2hz( self.mel_flist) # equal mel-frequency shifted  Hz frequency list
        #print (self.flist)

    def hz2mel(self,freq):
        return 2595. * np.log( freq / 700. + 1.0)
    
    def mel2hz(self,mel_freq):
        return 700. * ( np.exp( mel_freq / 2595) - 1.0)
    
    def get_postion(self, in_hzs):
        pos=[]
        cpos=[]
        for in_hz in in_hzs:
            pos.append( int((self.num_band-1.0) * (self.hz2mel( in_hz)-self.mel_fmin) / (self.mel_fmax - self.mel_fmin) ))
            cpos.append(str(in_hz))
        return pos,cpos
    


if __name__ == '__main__':
    
    from matplotlib import pyplot as plt
    from BPF4 import *
    
    # instance
    mel=Class_mel(num_band=1024)  #
    
    # set Q of BPF
    Q0=40.0
    sr0=44100 # sampling rate frequency
    
    # show mel-frequecny equal shifted  filter bank frequency response
    plt.xlabel('Hz')
    plt.ylabel('dB')
    plt.title('mel-frequency equal shifted Band Pass Filter bank')
    plt.xscale('log')
    for flist0 in mel.flist:
        print ( flist0 )
        bpf=Class_BPFtwice(fc=flist0, Q=Q0, sampling_rate=sr0)
        amp, freq= bpf.H0(freq_low=20, freq_high=10000, Band_num=1024)
        plt.plot(freq, amp)
        
    plt.grid()
    plt.tight_layout()
    plt.show()
