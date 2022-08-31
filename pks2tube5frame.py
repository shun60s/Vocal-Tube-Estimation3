#coding:utf-8

#
# Load wav (16Khz mono), LPC analysis, get peaks candidates and pitch(F0) candidate.
# Exclude outliers using a cubic expression and interpolate about peaks and pitch(F0).
# Estimate four tube model length and area by grid search and scipy's optimize.fmin, downhill simplex algorithm.
#
# argparse option:
#   --frame   use previous frame LA0 (estimated length and area) as initial value of scipy's optimize.fmin. Specify start frame number.
#   --one_frame   compute only one frame. Specify the frame number.
#   --BPF_out     compute BPF and show frequency response.
#


import sys
import os
import copy
import argparse
import numpy as np
from scipy import signal
from scipy import optimize
import matplotlib.pyplot as plt
import matplotlib.patches as patches


from tube_peak5 import *
from pre_compute5 import *
from curve_fit1 import *
from glottal import *
from HPF import *
from BPF_analysis2 import *


# Check version
# Python 3.10.4, 64bit on Win32 (Windows 10)
# numpy 1.22.3
# matplotlib  3.5.2
# scipy 1.8.0



def show_figure1(tube, peaks_target, drop_peaks_target, fmin0, LA0, F0=None, BPF_out=None, BPF_freq=None, path0=None):
    # comparison frequency response of tube with target
    
    NUM_TUBE= tube.NUM_TUBE
    
    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    if fmin0 >=0.:
        plt.title('frequency response: blue tube, green wav: min cost ' + str( round(fmin0,1)) )
    else:
        plt.title('frequency response: blue tube, green wav: X' )
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude [dB]')
    # tube spectrum
    tube_response_log= np.log10(tube.response) * 20
    ax1.plot(tube.f, tube_response_log, 'b', ms=2)
    ax1.plot(tube.f[tube.peaks_list] , tube_response_log[tube.peaks_list], 'ro', ms=3)
    if drop_peaks_target is not None:
        ax1.plot(tube.f[tube.drop_peaks_list] , tube_response_log[tube.drop_peaks_list], 'co', ms=3)
    
    xw= 2.0 * np.pi * peaks_target
    ax1.plot( peaks_target , np.log10(tube( LA0, xw_input=xw)) * 20 , 'x', ms=3)
    ax1.set_xlim(0,tube.f[-1])
    plt.grid()
    
    
    ax2 = fig.add_subplot(312)
    # BPF output
    if BPF_out is not None:
        if F0 is not None:
            # overall spectrum via tube model
            glo=Class_Glottal(F0=F0, sampling_rate=48000*4)
            glo_amp_repeat,_ =glo.H0_N_repeat(N_repeat=5, freq_list=tube.f)
            #print ('output',glo_amp_repeat)
            hpf=Class_HPF(sampling_rate=48000*4)
            hpf_response,_=hpf.H0(freq_list=tube.f)
            overall_response=glo_amp_repeat + tube_response_log + hpf_response
            #ax2.plot(tube.f, overall_response, 'y', label="overall") #, ms=2)
            overall_response2= (overall_response +100) * max(BPF_out)/ (max(overall_response)+100)
            ax2.plot(tube.f,overall_response2 , 'y', label='Tube overall')
            
        ax2.plot(BPF_freq, BPF_out, 'r', label="BPF out")
        
        ax2.set_ylim(bottom=0)
        ax2.set_xlim(0,tube.f[-1])
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Amplitude')
        plt.grid()
        plt.legend()
    
    
    ax3 = fig.add_subplot(313)
    if len(LA0) == 10 or len(LA0) == 9:  # X=[L1,L2,L3,L4,L5,A1,A2,A3,A4,A5] or X=[L1,L2,L3,L4,L5,r1,r2,r3,r4]   when five tube model
        L1= LA0[0]
        L2= LA0[1]
        L3= LA0[2]
        L4= LA0[3]
        L5= LA0[4]
        
        if len(LA0) == 10:
            A1= LA0[5]
            A2= LA0[6]
            A3= LA0[7]
            A4= LA0[8]
            A5= LA0[9]
        else:
            A1, A2, A3, A4, A5 = get_A1A2A3A4A5( LA0[5], LA0[6], LA0[7], LA0[8] )
        
        print ('L1,L2,L3,L4,L5', L1, L2, L3, L4, L5)
        print ('A1,A2,A3,A4,A5', A1, A2, A3, A4, A5)
    elif len(LA0) == 8 or len(LA0) == 7:  # X=[L1,L2,L3,L4,A1,A2,A3,A4] or X=[L1,L2,L3,L4,r1,r2,r3]   when four tube model
        L1= LA0[0]
        L2= LA0[1]
        L3= LA0[2]
        L4= LA0[3]
        L5= 0
        if len(LA0) == 8:
            A1= LA0[4]
            A2= LA0[5]
            A3= LA0[6]
            A4= LA0[7]
        else:
            A1, A2, A3, A4 = get_A1A2A3A4( LA0[4], LA0[5], LA0[6] )
        A5=0
        print ('L1,L2,L3,L4', L1, L2, L3, L4)
        print ('A1,A2,A3,A4', A1, A2, A3, A4)
    elif len(LA0) == 6 or len(LA0) == 5:  # X=[L1,L2,L3,A1,A2,A3] or X=[L1,L2,L3,r1,r2]   when three tube model
        L1= LA0[0]
        L2= LA0[1]
        L3= LA0[2]
        L4= 0
        L5= 0
        if len(LA0) == 6:
            A1= LA0[3]
            A2= LA0[4]
            A3= LA0[5]
        else:
            A1, A2, A3 = get_A1A2A3( LA0[3], LA0[4] )
        A4=0
        A5=0
        print ('L1,L2,L3', L1, L2, L3)
        print ('A1,A2,A3', A1, A2, A3)
        
        
    elif len(LA0) == 4 or len(LA0) == 3:  # L1,L2,r1 X=[L1,L2,A1,A2] or X=[L1,L2,r1] two tube model
        L1= LA0[0]
        L2= LA0[1]
        L3= 0
        L4= 0
        L5= 0
        if len(LA0) == 4:
            A1= LA0[2]
            A2= LA0[3]
        else:
            A1, A2 = get_A1A2( LA0[2] )
        A3=0
        A4=0
        A5=0
        print ('L1,L2', L1, L2)
        print ('A1,A2', A1, A2)
        
    ax3.add_patch( patches.Rectangle((0, -0.5* A1), L1, A1, hatch='/', fill=False))
    ax3.add_patch( patches.Rectangle((L1, -0.5* A2), L2, A2, hatch='/', fill=False))
    ax3.add_patch( patches.Rectangle((L1+L2, -0.5* A3), L3, A3, hatch='/', fill=False))
    ax3.add_patch( patches.Rectangle((L1+L2+L3, -0.5* A4), L4, A4, hatch='/', fill=False))
    ax3.add_patch( patches.Rectangle((L1+L2+L3+L4, -0.5* A5), L5, A5, hatch='/', fill=False))
    ax3.set_xlim([0, L1+L2+L3+L4+L5+5])
    ax3.set_ylim([(max(A1,A2,A3,A4,A5)*0.5+5)*-1, max(A1,A2,A3,A4,A5)*0.5+5 ])
    
    
    ax3.set_title('cross-section area')
    plt.xlabel('Length [cm]')
    plt.ylabel('Cross-section area [ratio]')
    plt.grid()
    plt.tight_layout()
    
    if path0 is not None:
        plt.savefig(path0)
        plt.clf()
        plt.close()
    else:
        plt.show()


def get_path_name( dir0, path0, number0, dir2=None,):
	# return file path name
	
    # make dir if the directory dir0 is not exist
    if not os.path.isdir( dir0 ):
        os.mkdir( dir0 )
    if (dir2 is not None)  and (not os.path.isdir( dir0 + '/' + dir2 )):
        os.mkdir( dir0 + '/' + dir2 )
    # get path0 basename without ext
    f, _= os.path.splitext( os.path.basename(path0))
    
    if dir2 is not None:
        return dir0 + '/' +  dir2 + '/' + f + '_' + str(number0) + '.png'
    else:
        return dir0 + '/' + f + '_' + str(number0) + '.png'


if __name__ == '__main__':
    #
    parser = argparse.ArgumentParser(description='load wav file and estimate tube model ')
    parser.add_argument('--wav_file', '-w', default='wav/a_1-16k.wav', help='specify input wav-file-name(mono,16bit,16Khz)')
    parser.add_argument('--tube',  '-t', type=int, default=4, help='specify number of tube, 2 or 3 or 4 or 5')
    parser.add_argument('--result_dir', '-r', default='result_figure', help='specify result directory')
    parser.add_argument('--BPF_out','-B', action='store_true', help='show BPF output')
    parser.add_argument('--frame', '-f', type=int, default=-1, help='specify start frame number, igonred if negative')
    parser.add_argument('--one_frame', '-o', type=int, default=-1, help='compute only one frame. specify the frame number, igonred if negative')
    args = parser.parse_args()
    
    # sub directory control
    # set 1 if there is pks_dpks_stack_tube_use_ratio xxx.npz in SUB_DIR_PATH.
    if 1:
        SUB_DIR_PATH='pks_dpks_stack/'
    else:
        SUB_DIR_PATH=''
    
    NUM_TUBE = args.tube 
    # instance tube model
    tube= compute_tube_peak(NUM_TUBE=NUM_TUBE)  #, disp=True)
    # load pre-computed grid data
    pc1=pre_comute(tube, path0=  SUB_DIR_PATH + 'pks_dpks_stack_tube_use_ratio' + str(NUM_TUBE) + '.npz')
    
    # get curve fit value as peaks
    CF1= Class_Curve_Fit1( args.wav_file, SHOW=True)
    
    # compute BPF output
    if args.BPF_out:
        Ana1= Class_Analysis1(num_band=1024, fmin=40, fmax=6000, sr=CF1.sr)
        yo= Ana1.compute(CF1.fdata2)
    
    # ignore drop_peaks_target
    drop_peaks_target=None 
    
    
    if args.frame >= 0:
        frame_list= range(args.frame, len( CF1.peak_list_new))
        frame_list2= range(args.frame,-1,-1)
        frame_lists=[frame_list, frame_list2]
        dir2=str(args.frame)
    elif args.one_frame >= 0:  # compute one frame only
        frame_lists=[[ args.one_frame] ] 
        dir2=str(args.one_frame)
    else:
        frame_list= range(len( CF1.peak_list_new))
        frame_lists=[frame_list]
        dir2=None
    
    for list0 in frame_lists:
        next_X=None
        for l, nframe in enumerate(list0):
            print ('-frame number', nframe)
            # set expect target value
            peak_list= CF1.peak_list_new[nframe]
            peaks_target=peak_list[0:NUM_TUBE]
            F0= CF1.pout_new[nframe]
            
            if args.BPF_out:
                postion_center=CF1.NSHIFT_time * nframe + CF1.NFRAME_time/2
                index_of_postion_center= int((postion_center/1000) * (CF1.sr/Ana1.dsf))
                BPF_out= yo[:,index_of_postion_center]
                BPF_freq=Ana1.mel.flist
            else:
                BPF_out=None
                BPF_freq=None
            
            if next_X is None:
                # get minimun cost at grid   
                X, grid_peaks = pc1.get_min_cost_candidate(peaks_target,drop_peaks_target, symmetry=True, disp=False, grid_peaks=True)
            else:
                X= next_X
                
            # try to minimize the function
            #   by "fmin" that is minimize the function using the downhill simplex algorithm.
            args1=(peaks_target,drop_peaks_target, -1)
            
            # xtol, ftolを調整していく。　計算時間は長くなるが、誤差は小さくなっていくようだ。
            if NUM_TUBE == 5:
                res_brute = optimize.fmin( tube.calc_cost, X, args=args1, xtol=0.00001, ftol=0.00001,full_output=True, disp=False)  #xtol=0.0001, ftol=0.0001,
            else:
                res_brute = optimize.fmin( tube.calc_cost, X, args=args1,full_output=True, disp=False)
            
            print ( 'min cost %f LA ' % (res_brute[1]) , res_brute[0] ) 
            #print ( 'minimum ', res_brute[0] )  # minimum
            #print ( 'function value ', res_brute[1] )  # function value at minimum
            if res_brute[4] != 0:  # warnflag
                print ('warnflag is not 0')
            
            tube(res_brute[0]) 
            path0=get_path_name( args.result_dir, args.wav_file, nframe, dir2=dir2)
            show_figure1(tube, peaks_target, drop_peaks_target, res_brute[1], res_brute[0], F0=F0, BPF_out=BPF_out, BPF_freq=BPF_freq, path0=path0)
            
            # use previous frame LA0 (estimated length and area) as initial value of next
            if args.frame >= 0:
                next_X=res_brute[0].copy()
        