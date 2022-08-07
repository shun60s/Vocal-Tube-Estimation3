#coding:utf-8

# BPF bank analysis Spectrogram
#
#  feature
#   BPF's target response is 2nd harmonic level less than -70dB
#   Mel-frequency division
#   Half-wave rectification until a few KHz signal or DC with ripple signal
#   Down sampling to decrease temporal resolution
#   N-th root compression 
#   simple moving average until 800Hz signal  *major change from BPF_ana1.py
#   normalized Gray scale image output

import sys
import copy
import argparse
from scipy import signal
from scipy.signal import find_peaks
from scipy import interpolate
from scipy import optimize
from scipy.io.wavfile import read as wavread
from matplotlib import pyplot as plt

from mel  import *
from BPF4 import *
from Compressor1 import *
from iir1 import *


# Check version
#  Python 3.10.4 on win32 (Windows 10)
#  numpy 1.21.6
#  scipy 1.8.0
#  matplotlib  3.5.2

class Class_Analysis1(object):
    def __init__(self, num_band=1024, fmin=40, fmax=8000, sr=44100, Q=40.0, \
        moving_average_factor=50, down_sample_factor=10, \
        power_index=1/3.5, SMA=True, SMA_max_freq=800):
        # instance
        # (1) mel frequency list
        self.num_band=num_band
        self.fmin=fmin
        self.fmax=fmax
        self.mel=Class_mel(self.num_band, self.fmin, self.fmax)
        self.freq_linear = np.linspace(self.mel.flist[0],self.mel.flist[-1],num=int(self.mel.flist[-1]-self.mel.flist[0]),endpoint=True)
        
        # (2) BPF bank
        self.sr= sr
        self.Q= Q
        self.maf= int(moving_average_factor)
        self.dsf= int(down_sample_factor)
        self.BPF_list=[]
        for flist0 in self.mel.flist:
            bpf=Class_BPFtwice(fc=flist0, Q=self.Q, sampling_rate=self.sr, moving_average_factor=self.maf, down_sample_factor=self.dsf)
            self.BPF_list.append(bpf)
        # (3) compress via power function
        self.power_index= power_index
        self.comp1= Class_Compressor1(power_index= self.power_index)
        
        # (4) simple moving average until SMA_max_freq signal
        self.SMA_ON= SMA
        self.SMA_max_freq=SMA_max_freq
        
    def compute(self, yg):
        # yg should be mono
        self.dwn_len= int(len(yg)/self.dsf)
        self.out1= np.empty( ( self.num_band, self.dwn_len), dtype=np.float32  )
        
        for i, bpf in enumerate( self.BPF_list ):
            print ('\r fc', bpf.fc, end='')
            self.out1[i]=self.comp1(bpf.filtering2( yg, self.dwn_len))
        print ('')
        
        if self.SMA_ON:
            print('simple moving average')
            self.out1=self.simple_moving_average()
        
        print ('self.out1.shape', self.out1.shape)
        print ('max', np.amax(self.out1), ' min', np.amin(self.out1))
        
        return self.out1
        
    def simple_moving_average(self, PLOT_SHOW=False):
        index0= np.where(self.mel.flist >= self.SMA_max_freq)[0][0]
        self.out2=self.out1.copy()
        for i in range(index0+1):
            w= int(self.sr / self.dsf / self.mel.flist[i])
            if w > 1:
                self.out2[i,:]= np.convolve(np.concatenate([np.zeros(w-1),self.out1[i,:]]), np.ones(w), 'valid') / w
                
                if PLOT_SHOW:
                    fig = plt.figure()
                    ax1 = fig.add_subplot(211)
                    ax1.plot(self.out1[i,:], 'r', label="BPF out")
                    ax1.plot(self.out2[i,:], 'y', label="SMA out")
                    plt.ylabel('Amplitude')
                    plt.grid()
                    plt.legend()
                    plt.tight_layout()
                    plt.show()
            else:
                print ('warning: w <= 1, simple moving average',self.mel.flist[i])
        return self.out2
    
    def trans_gray(self, indata0 ):
        # in_data0 dimension should be 2 zi-gen
        # convert to single Gray scale
        f= np.clip( indata0, 0.0, None)  # clip to >= 0
        # Normalize to [0, 255]
        f=  f / np.amax(f)  # normalize as max is 1.0
        fig_unit = np.uint8(np.around( f * 255))
        return fig_unit
    
    def conv_gray2RGBgray(self, in_fig ):
        # convert single Gray scale to RGB gray
        rgb_fig= np.zeros( (in_fig.shape[0],in_fig.shape[1], 3) )
        
        for i in range(3):
            rgb_fig[:,:,i] = 255 - in_fig
        
        return rgb_fig
    
    def conv_int255(self, in_fig):
        # matplotllib imshow x format was changed from version 2.x to version 3.x
        if 1:  # matplotlib > 3.x
            return np.array(np.abs(in_fig - 255), np.int32)
        else:  # matplotlib = 2.x
            return in_fig
    
    def plot_image(self, yg=None):
        #
        fig_image= self.conv_gray2RGBgray( self.trans_gray(self.out1))
        # 
        if yg is not None:
            fig,  [ax0, ax] = plt.subplots(2, 1)
            ax0.plot(yg)
            ax0.set_xlim(0, len(yg))
        else:
            fig, ax = plt.subplots()
        
        ax.set_title('BPF bank analysis Spectrogram')
        ax.set_xlabel('time step [sec]')
        ax.set_ylabel('frequecny [Hz]')
        
        # draw time value
        xlen=fig_image.shape[1]
        slen=xlen / ( self.sr/ self.dsf)
        char_slen=str( int(slen*1000) / 1000) # ms
        char_slen2=str( int((slen/2)*1000) / 1000) # ms
        ax.set_xticks([0,int(xlen/2)-1, xlen-1])
        ax.set_xticklabels(['0', char_slen2, char_slen])
        
        # draw frequecny value
        ylen=fig_image.shape[0]
        flens=[self.fmin, 100, 200, 300, 500,700, 1000,1500, 2000, 2500, 3000, 3500, 4000, 5000,6000, self.fmax]
        # flens=[self.fmin, 300, 600, 1000, 1400, 2000, 3000,  self.fmax] # forMix_400Hz1KHz-10dB_44100Hz_400msec_TwoTube_mono.wav
        yflens,char_flens= self.mel.get_postion( flens)
        ax.set_yticks( yflens )
        ax.set_yticklabels( char_flens)
        
        ax.imshow( self.conv_int255(fig_image), aspect='auto', origin='lower')
        
        plt.tight_layout()
        plt.show()


#-------------------------------------------------------
# class inheritance to include some helper functions
#  
#
# feature
#    tuning F0 to maximize sum of harmonic strength
#    frequency response estimation by curve fitting via F0 harmonic frequencies (fundamental and overtones) 
#

class Class_Analysis2(Class_Analysis1):
    def __init__(self, num_band=1024, fmin=40, fmax=8000, sr=44100, Q=40.0, \
                  moving_average_factor=50, down_sample_factor=10, \
                  power_index=1/3.5, SMA=True,SMA_max_freq=800, nframe_time=25, nshift_time=10):
        super().__init__(num_band, fmin, fmax, sr, Q, \
                         moving_average_factor, down_sample_factor, \
                         power_index, True, SMA_max_freq)
        
        # (5) frame
        self.nframe_time= nframe_time  # unit [ms]
        self.nshift_time= nshift_time  # unit [ms]
        self.sr_new=  self.sr / self.dsf
        self.nframe= int(self.sr_new * self.nframe_time / 1000)
        self.nshift= int(self.sr_new * self.nshift_time / 1000)
        
        # (6)
        self.max_num_peaks=5
        self.fout=None
        self.pout=None
    
    #^^~ override compute
    def compute(self, yg):
        # yg should be mono
        self.dwn_len= int(len(yg)/self.dsf)
        self.out1= np.empty( ( self.num_band, self.dwn_len), dtype=np.float32  )
        
        for i, bpf in enumerate( self.BPF_list ):
            print ('\r fc', bpf.fc, end='')
            self.out1[i]=self.comp1(bpf.filtering2( yg, self.dwn_len))
        print ('')
        
        if self.SMA_ON:
            print('simple moving average')
            self.out1=self.simple_moving_average()
        
        print ('self.out1.shape', self.out1.shape)
        print ('max', np.amax(self.out1), ' min', np.amin(self.out1))
        
        self.frames= int(((self.out1.shape[1] - self.nframe) / self.nshift) + 1)
        print ('number of frames', self.frames)
        self.frames_center= np.array(np.linspace(0, self.frames-1, self.frames) * self.nshift + (self.nframe/2), np.int32)
        #print ('frame center', self.frames_center)
        self.fout=np.zeros([self.frames,self.max_num_peaks])
        self.pout=np.zeros(self.frames)
        
        return self.out1
    
    def compute2(self,F0=80, PLOT_SHOW=False):
        for l, pos in enumerate( self.frames_center):
            # skip
            #if l != 7: # and l != 14 :
            #    continue
            # change from mel scale to linear
            func1 = interpolate.interp1d(self.mel.flist,self.out1[:,pos] , kind="cubic")
            fout1 = func1(self.freq_linear)
            peaks, _ = find_peaks(fout1, distance= F0 * 0.9,  prominence= max(fout1) * 0.1 )
            if len(peaks) > 0:
                self.pout[l]= self.freq_linear[peaks[0]]
                F0_new=self.pout[l]
            else:
                F0_new=F0
            
            #--- tuning F0
            # 1st step until 7 times harmonic
            self.funcx=func1
            self.numberx=7
            rranges= [(F0_new * 0.9,  F0_new * 1.1)]
            resbrute = optimize.brute( self.cost,rranges )
            #print ('1st step. min F0 via optimize.brute. F0_new', resbrute[0])
            F0_new= resbrute[0]
            
            # 2nd step  until 2kHz
            upper_freq=2000
            self.numberx=int(upper_freq / F0_new)
            rranges= [(F0_new * 0.95,  F0_new * 1.05)]
            resbrute = optimize.brute( self.cost,rranges )
            #print ('2nd step. min F0 via optimize.brute. F0_new', resbrute[0])
            F0_new= resbrute[0]
            
            # last, just compute harmonic frequencies until 6kHz
            upper_freq=6000
            self.number=int(upper_freq / F0_new)
            self.hamonic_freq_list=np.linspace( F0_new, F0_new * self.number, self.number)
            fout1_harmonic= func1(self.hamonic_freq_list)
            
            # set final candiate F0_new as pout
            self.pout[l]= F0_new
            #--- end of tuning F0
            
            # get index of the range
            p0=np.where(self.freq_linear > F0_new )[0]
            p1=np.where(self.freq_linear > F0_new * self.number )[0]
            
            # curve fitting via harmonic frequencies
            func2 = interpolate.interp1d( self.hamonic_freq_list,fout1_harmonic , kind="cubic")
            fout2 = func2(self.freq_linear[ p0[0]:p1[0]])
            
            #-- peak search
            # try 1
            thres0=0.05
            peak_curve_peaks, _ = find_peaks(fout2,distance= F0_new, prominence= max(fout2) * thres0 )
            # try 2 when there are not enough candiates
            if len(peak_curve_peaks) < self.max_num_peaks-1:
                thres0=thres0 * 0.75
                peak_curve_peaks, _ = find_peaks(fout2,distance= F0_new, prominence= max(fout2) * thres0 )
            
            if len(peak_curve_peaks) > 0:
                idm= min(len(peak_curve_peaks),self.max_num_peaks)
                self.fout[l,0:idm]=self.freq_linear[peak_curve_peaks[0:idm] + p0[0]]
            
            
            print ('-no. frame',l)
            print ('fout(peaks)', self.fout[l,:],' pout', self.pout[l])
            
            if PLOT_SHOW:
                fig = plt.figure()
                ax1 = fig.add_subplot(211)
                ax1.plot(self.mel.flist,self.out1[:,pos], 'r', label="BPF out")
                ax1.plot(self.freq_linear, fout1, 'y', label="interpolate")
                ax1.plot(self.hamonic_freq_list ,fout1_harmonic , 'x', ms=3, label="F0 harmonic")
                ax1.plot(self.freq_linear[ p0[0]:p1[0]], fout2, 'm', label="curve fitting")
                ax1.plot(self.freq_linear[peak_curve_peaks+p0[0]] , fout2[peak_curve_peaks] , 'o', ms=3, label="peak")
                
                plt.xlabel('Frequency [Hz]')
                plt.ylabel('Amplitude')
                plt.grid()
                plt.legend()
                
                plt.tight_layout()
                plt.show()
            
    def cost(self, F0):
        # compute sum of harmonic strength, to search F0 which maximize it
        # 高調波成分の和を計算する。高調波成分の和が最大になるF0を探す。
        hamonic_freq_listx=np.linspace( F0, F0 * self.numberx, self.numberx)
        return -1. * np.sum( self.funcx(hamonic_freq_listx) )
        
        
    def show_one_channel(self,freq_show, freq_show2=None):
        index0= np.where(self.mel.flist >= freq_show)[0][0]
        print ('mel freq', self.mel.flist[index0])
        if freq_show2 is not None:
            index2= np.where(self.mel.flist >= freq_show2)[0][0]
            print ('mel freq2', self.mel.flist[index2])
        
        if 1:
            fig = plt.figure()
            ax1 = fig.add_subplot(211)
            
            ax1.plot(self.out1[index0,:] , 'r', label= str(self.mel.flist[index0]))
            ax1.plot(self.out1[index0+1,:] , 'b', label= str(self.mel.flist[index0+1]))
            plt.ylabel('Amplitude')
            plt.grid()
            plt.legend()
            
            if freq_show2 is not None:
                ax2 = fig.add_subplot(212)
                
                ax2.plot(self.out1[index2,:] , 'r', label= str(self.mel.flist[index2]))
                ax2.plot(self.out1[index2+1,:] , 'b', label= str(self.mel.flist[index2+1]))
                plt.ylabel('Amplitude')
                plt.grid()
                plt.legend()
            
            
            plt.tight_layout()
            plt.show()
    
#
#--------------------------------------------------------


def load_wav( path0):
    # return 
    #        yg: wav data (mono) 
    #        sr: sampling rate
    try:
        sr, y = wavread(path0)
    except:
        print ('error: wavread ', path0)
        sys.exit()
    else:
        yg= y / (2 ** 15)
        if yg.ndim == 2:  # if stereo
            yg= np.average(yg, axis=1)
    
    print ('file ', path0)
    print ('sampling rate ', sr)
    print ('length ', len(yg))
    return yg,sr



if __name__ == '__main__':
    #
    parser = argparse.ArgumentParser(description='BPF bank analysis Spectrogram')
    parser.add_argument('--wav_file', '-w', default='wav/a_1-16k.wav', help='wav file name(16bit)')
    parser.add_argument('--nframe_time', '-f', type=int, default=25, help='specify one frame time [ms]')
    parser.add_argument('--nshift_time', '-s', type=int, default=10, help='specify shift time [ms]')
    args = parser.parse_args()
    
    # load wav file
    path0= args.wav_file
    yg,sr=load_wav( path0)
    
    """
    # instance
    Ana1= Class_Analysis1(num_band=1024, fmin=40, fmax=8000, sr=sr)
    # process
    yo= Ana1.compute(yg)
    # draw image
    Ana1.plot_image()
    """
    
    # instance
    Ana2= Class_Analysis2(num_band=1024, fmin=40, fmax=8000, sr=sr, nframe_time=args.nframe_time, nshift_time=args.nshift_time)
    # process BPF
    yo= Ana2.compute(yg)
    # draw image
    Ana2.plot_image()
    # process to get fout(peaks) and pout
    Ana2.compute2(PLOT_SHOW=True)
    
   
