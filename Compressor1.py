#coding:utf-8

# A class of compressor by power(input,1/3.5), 3.5 root compression 

import numpy as np

# Check version
#  Python 3.6.4 on win32 (Windows 10)
#  numpy 1.16.3
#  matplotlib  2.1.1
#  scipy 1.4.1

class Class_Compressor1(object):
    def __init__(self, Vt=1.0, power_index=1/3.5, Clip_bottom=1.0e-7):  
        # initalize
        self.Vt= Vt
        self.power_index= power_index
        self.Clip_bottom= Clip_bottom

    def __call__(self, x):
        #   power( clip(x, Clip_bottom), power_index) * Vt
        #
        return self.Vt * np.power( np.clip(x, self.Clip_bottom, None), self.power_index )
    
    
if __name__ == '__main__':
    from scipy import signal
    from scipy.io.wavfile import read as wavread
    from matplotlib import pyplot as plt
    
    from BPF4 import *
    
    # instance
    fc1=2000
    dsf=10
    bpf=Class_BPFtwice(fc=fc1,  Q=10.0, sampling_rate=44100, moving_average_factor=80, down_sample_factor=dsf)
    
    
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
        if yg.ndim == 2:  # if stereo
            yg= np.average(yg, axis=1)
        print ('sampling rate ', sr)
        print ('y.shape', yg.shape)
    
    y5=bpf.filtering2(yg, int(len(yg)/dsf))
    
    comp1=Class_Compressor1(power_index=1/2)
    # compute
    y6= comp1(y5)
    
    bpf.wav_show(yg,y5,y6)
