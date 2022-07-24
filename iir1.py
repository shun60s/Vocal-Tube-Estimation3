#coding:utf-8

# A class of iir butterworth filter

# Check version
#  Python 3.6.4 on win32 (Windows 10)
#  numpy 1.16.3
#  scipy 1.4.1
#  matplotlib  2.1.1


import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


class Class_IIR1(object):
    def __init__(self, fc=2205.0, btype='low', n_order=2, sampling_rate=44100):
        # design iir butterworth filter
        # initalize
        self.fc= fc # cut off frequency of High Pass Filter by unit is [Hz]
        self.sr= sampling_rate
        self.n_order= n_order
        self.btype= btype
        self.b, self.a= signal.butter(self.n_order, (self.fc / (self.sr/2.0)) , btype=self.btype, analog=False, output='ba') # b=Numerator(bunsi), a=denominator(bunbo)

    def filtering(self, xin):
        return signal.lfilter(self.b, self.a, xin)

    def f_show(self, worN=1024):
        # show frequency response
        wlist, fres = signal.freqz(self.b, self.a, worN=worN)
        
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

    def N(self,):
        # if moving average is approximated as a LPF
        # n is span of moving average
        n=np.sqrt(np.power( 0.443 * self.sr / self.fc, 2.0) +1.0) 
        print (n)
        
if __name__ == '__main__':
    
    # instance
    sampling_rate=44100
    fc=sampling_rate / 20  # re-sample sampling_rate / 10 ?
    lpf=Class_IIR1(fc=fc, sampling_rate=sampling_rate)
    lpf.f_show()
    
    lpf.N()
    



