#coding:utf-8

# BPF bank analysis Spectrogram
#
#  feature
#   BPF's target response is 2nd harmonic level less than -70dB
#   Mel-frequency division
#   Half-wave rectification until a few KHz signal or DC with ripple signal
#   Down sampling to decrease temporal resolution
#   N-th root compression 
#   normalized Gray scale image output

import sys
import argparse
from scipy import signal
from scipy.io.wavfile import read as wavread
from matplotlib import pyplot as plt

from mel  import *
from BPF4 import *
from Compressor1 import *


# Check version
#  Python 3.6.4 on win32 (Windows 10)
#  numpy 1.18.4
#  matplotlib  3.3.1
#  scipy 1.4.1

class Class_Analysis1(object):
    def __init__(self, num_band=1024, fmin=40, fmax=8000, sr=44100, Q=40.0, \
        moving_average_factor=50, down_sample_factor=10, \
        power_index=1/3.5):
        # instance
        # (1) mel frequency list
        self.num_band=num_band
        self.fmin=fmin
        self.fmax=fmax
        self.mel=Class_mel(self.num_band, self.fmin, self.fmax)
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
        
    def compute(self, yg):
        # yg should be mono
        self.dwn_len= int(len(yg)/self.dsf)
        self.out1= np.empty( ( self.num_band, self.dwn_len), dtype=np.float32  )
        
        for i, bpf in enumerate( self.BPF_list ):
            print ('\r fc', bpf.fc, end='')
            self.out1[i]=self.comp1(bpf.filtering2( yg, self.dwn_len))
        
        print ('self.out1.shape', self.out1.shape)
        print ('max', np.amax(self.out1), ' min', np.amin(self.out1))
        
        return self.out1
    
    
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
    parser.add_argument('--wav_file', '-w', default='wav/1KHz-10dB_44100Hz_400ms-TwoTube_stereo.wav', help='wav file name(16bit)')
    args = parser.parse_args()
    
    path0= args.wav_file
    # overwrite wav file name
    #path0="wav/mix_a_1st_frame_head4.wav"
    #path0='wav2/Mix_400Hz1KHz-10dB_44100Hz_400msec_MONO.wav'
    #path0='wav2/400Hz-10dB_44100Hz_400msec.wav'
    #path0='wav2/1KHz-10dB_44100Hz_400msec.wav'
    #path0='wav2/3KHz-10dB_44100Hz_400msec.wav'
    #path0='wav2/5KHz-10dB_44100Hz_400msec.wav'
    #path0='wav2/2KHz-80dB_44100Hz_400msec.wav'
    #path0='wav2/1KHz-10dB_44100Hz_400ms-TwoTube_stereo.wav'
    #path0='wav2/Mix_400Hz1KHz-10dB_44100Hz_400msec_TwoTube_mono.wav'
    
    yg,sr=load_wav( path0)
    
    # instance
    Ana1= Class_Analysis1(num_band=1024, fmin=40, fmax=8000, sr=sr)
    
    # process
    yo= Ana1.compute(yg)
    
    # draw image
    Ana1.plot_image()
    #Ana1.plot_image(yg)
