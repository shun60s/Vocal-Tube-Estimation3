#coding:utf-8

#
# get_fp6: LPC分析によるピーク周波数（ホルマント）とピッチ（F0)の推定値の中から、
# 外れ値を除外して、３次式で近似した値を返す
#

import numpy as np
from scipy.optimize import curve_fit  
from matplotlib import pyplot as plt
from get_fp6 import *

# Check version
#  Python 3.6.4 on win32 (Windows 10)
#  numpy 1.18.4
#  matplotlib  3.3.1
#  scipy 1.4.1

class Class_Curve_Fit1(object):
    def __init__(self, wav_file, SHOW=False):
        #
        self.wav_file=wav_file
        # instance get_fp6
        self.fp6= Class_get_fp(f_min=200)
        self.NSHIFT_time= self.fp6.NSHIFT_time
        self.NFRAME_time= self.fp6.NFRAME_time 
        
        # load wav and get formant candidates
        self.peak_list0, drop_peak_list0, spec_out, fout_index, fout_index2, self.pout= self.fp6.get_fp(self.wav_file, frame_num=-1)
        print ('number of frames ', len( self.peak_list0 ))
        #print ('peak_list0.shape', self.peak_list0.shape)
        self.sr    = self.fp6.sr
        self.fdata2= self.fp6.fdata2
        #
        peak1=self.peak_list0[:,0]
        peak2=self.peak_list0[:,1]
        peak3=self.peak_list0[:,2]
        peak4=self.peak_list0[:,3]
        peak5=self.peak_list0[:,4]
        peak1_new, _= self.curve_fit_with_iqr(peak1)
        peak2_new, _= self.curve_fit_with_iqr(peak2)
        peak3_new, _= self.curve_fit_with_iqr(peak3)
        peak4_new, _= self.curve_fit_with_iqr(peak4)
        peak5_new, _= self.curve_fit_with_iqr(peak5)
        
        # self.pout_new and self.peak_list_new is output, curve fit data
        self.pout_new, self.t_step= self.curve_fit_with_iqr(self.pout, show=False)
        self.peak_list_new= np.vstack([peak1,peak2,peak3,peak4,peak5]).T
        #print ('self.peak_list_new.shape',self.peak_list_new.shape)
        
        if SHOW:
            fig = plt.figure()
            ax1 = fig.add_subplot(211)
            ax1.plot(self.t_step, peak1, label='peak1')
            ax1.plot(self.t_step, peak1_new,'x')
            ax1.plot(self.t_step, peak2, label='peak2')
            ax1.plot(self.t_step, peak2_new,'x')
            ax1.plot(self.t_step, peak3, label='peak3')
            ax1.plot(self.t_step, peak3_new,'x')
            ax1.plot(self.t_step, peak4, label='peak4')
            ax1.plot(self.t_step, peak4_new,'x')
            ax1.plot(self.t_step, peak5, label='peak5')
            ax1.plot(self.t_step, peak5_new,'x')
            plt.title(self.wav_file)
            plt.grid()
            plt.legend()
           
            ax2 = fig.add_subplot(212)
            ax2.plot(self.t_step, self.pout, 'r', label='pout(F0)')
            ax2.plot(self.t_step, self.pout_new, 'y', label='curve_fit')
            plt.grid()
            plt.legend()
            
            plt.tight_layout()
            plt.show()
            

    def iqr1(self,xin, max_diff_time=2, min_value=1, show_dump=False):
        # 四分位範囲Interquartile rangeの方法で除外する
        # centerのmax_diff_time倍より大きいものも除外する
        # min_vale以下のものを除外する
        # 外れ値を除く、有効なindexを返す
        xin_sort=np.sort(xin)
        Q1=int(len(xin)*0.25)
        Q2=int(len(xin)*0.5)
        Q3=int(len(xin)*0.75)
        iqr=xin_sort[Q3] - xin_sort[Q1]
        center=xin_sort[Q2]
        Out_min=xin_sort[Q1] - 1.5 * iqr
        Out_max=xin_sort[Q3] + 1.5 * iqr
        in_index=np.where( (xin > Out_min) & (xin < Out_max)  & (xin <center *max_diff_time) & (xin > min_value))
        
        if show_dump:
            print('xin',xin)
            print('xin_sort',xin_sort)
            print('Out_min,Out_max', Out_min,Out_max)
            print('xin[in_index]', xin[in_index])
        
        return in_index
    
    def curve_fit_with_iqr(self,xin, show=False):
        #　外れ値を除外して、3次式で近似した値を返す。
        in_index=self.iqr1(xin, show_dump=show)
        t_steps=np.linspace(0, len(xin),len(xin))
        popt, pcov = curve_fit(self.func_3,t_steps[in_index],xin[in_index])
        xin_new=self.func_3(t_steps, popt[0],popt[1],popt[2],popt[3])
        return xin_new, t_steps
    
    def func_3(self, X, a, b, c, d): 
        # 3次式近似
        Y = a + b * X + c * X ** 2 + d * X ** 3
        return Y


if __name__ == '__main__':
    #
    a1= Class_Curve_Fit1('wav/a_1-16k.wav', SHOW=True)
    i1= Class_Curve_Fit1('wav/i_1-16k.wav', SHOW=True)
    u1= Class_Curve_Fit1('wav/u_1-16k.wav', SHOW=True)
    e1= Class_Curve_Fit1('wav/e_1-16k.wav', SHOW=True)
    o1= Class_Curve_Fit1('wav/o_1-16k.wav', SHOW=True)
