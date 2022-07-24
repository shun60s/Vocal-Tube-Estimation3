#coding:utf-8

# precompute value at grid of LA_ranges and save as a npz file, Or load precomputed data from a npz file
#
# This version is using frequency ratio.

import sys
import argparse
import itertools
import numpy as np


# Check version
# Python 3.10.4, 64bit on Win32 (Windows 10)
# numpy 1.22.3

class pre_comute(object):
    def __init__(self, tube, LA_ranges=None, MAX_tube_length=30., path0=None, display_count=100, USE_COST_RATIO=True):
        #
        self.tube= tube
        self.NUM_TUBE= tube.NUM_TUBE
        self.display_count= display_count
        self.USE_COST_RATIO=USE_COST_RATIO  # 周波数の比である2番目以降の要素を使って計算する。
        
        # compute when LA_ranges is specified
        if LA_ranges is not None:
            self.LA_ranges= LA_ranges
            self.MAX_tube_length= MAX_tube_length  # specify maximum whole tube length
            self.compute()
            # save compute data to path0
            if path0 is not None:  # tube_stack.npz
                np.savez(path0, pks=self.peaks_detail_stack, dpks=self.drop_peaks_detail_stack,\
                 iter=self.iters_stack, misc=[self.MAX_tube_length] )
                print ('pre compute data was saved to ', path0)
        
        # load pre-computed data from path0
        elif path0 is not None:
            try:
                x=np.load(path0)
            except:
                print ('error: cannot load ', path0)
                sys.exit()
            else:
                self.peaks_detail_stack=x['pks']
                self.drop_peaks_detail_stack=x['dpks']
                self.iters_stack=x['iter']
                self.MAX_tube_length=x['misc'][0]
                # check
                self.iters_len = self.peaks_detail_stack.shape[0]
                if self.peaks_detail_stack.shape[1] != self.NUM_TUBE:
                    print ('error: NUM_TUBE is mismatch.')
                    sys.exit()
                else:
                    print ('pre compute data was loaded from ', path0, self.iters_len)
        else:
            print ('error: no specified inputs.')
            sys.exit()
            
    def compute(self,):
        # slice数が大きいとoverして動かなくなるので、LとAとに分割する。
        L_len= int( (len(self.LA_ranges)+1)/2)
        LA_L_ranges= self.LA_ranges[0 : L_len]
        LA_A_ranges= self.LA_ranges[L_len:]
        
        itersL=[]
        for slice0 in LA_L_ranges:
            # add slice stop value in iteration
            itersL.append( np.append( np.arange( slice0.start, slice0.stop, slice0.step), slice0.stop) )
        
        itersA=[]
        for slice0 in LA_A_ranges:
            # add slice stop value in iteration
            itersA.append( np.append( np.arange( slice0.start, slice0.stop, slice0.step), slice0.stop) )
        
        
        cL=0
        for i, paramL in enumerate( itertools.product(*itersL)):
            if self.USE_COST_RATIO:
                if np.array(paramL[0:]).sum() != self.MAX_tube_length: # count MAX_tube_length equal case
                    continue
            else:
                if np.array(paramL[0:2]).sum() > self.MAX_tube_length: # skip whole tube length is over than max_tube_length
                    continue
            cL=cL+1
        
            cA=0
            for i, paramA in enumerate( itertools.product(*itersA)):
                cA=cA+1
        
        self.iters_len= cL * cA  # 実際のitertoolの数
        print ('iteraion number ', self.iters_len)
        
        self.peaks_detail_stack= np.zeros((self.iters_len, self.NUM_TUBE))
        self.drop_peaks_detail_stack= np.zeros((self.iters_len, self.NUM_TUBE))
        self.iters_stack= np.zeros( (self.iters_len, len(self.LA_ranges)) )
        
        
        c0=0
        c0_skip_mode=0
        for iL, paramL in enumerate( itertools.product(*itersL)):
            
            if self.USE_COST_RATIO:
                if np.array(paramL[0:]).sum() != self.MAX_tube_length: # skip MAX_tube_length not equal case
                    continue
            else:
                if np.array(paramL[0:2]).sum() > self.MAX_tube_length: # skip whole tube length is over than max_tube_length
                    continue
            
            for iA, paramA in enumerate( itertools.product(*itersA)):
                #
                param0= np.append(paramL, paramA)
                # ピーク数がNUM_TUBE未満のデータは除外する。
                peaks_detail, drop_peaks_detail= tube( param0 )
                if peaks_detail is None:
                    c0_skip_mode +=1
                    if (c0 > 0 and (c0+c0_skip_mode) % self.display_count == 0) or c0 == (self.iters_len-1):
                        sys.stdout.write("\r%d (%d)" % (c0, c0+c0_skip_mode) )
                        sys.stdout.flush()
                    continue
                
                if self.USE_COST_RATIO:
                    # 2番目以降は、1番目の周波数からの比率を入れる
                    self.peaks_detail_stack[c0]= peaks_detail/peaks_detail[0]
                    self.peaks_detail_stack[c0,0]=peaks_detail[0]
                    self.drop_peaks_detail_stack[c0]= drop_peaks_detail/drop_peaks_detail[0]
                    self.drop_peaks_detail_stack[c0,0]= drop_peaks_detail[0]
                else:
                    self.peaks_detail_stack[c0]= peaks_detail
                    self.drop_peaks_detail_stack[c0]= drop_peaks_detail
                
                self.iters_stack[c0]= np.array( param0)
                c0 +=1
                
                if (c0 > 0 and (c0+c0_skip_mode) % self.display_count == 0) or c0 == (self.iters_len-1):
                    sys.stdout.write("\r%d (%d)" % (c0, c0+c0_skip_mode) )
                    sys.stdout.flush()
        print ("")
        if c0 != self.iters_len:
            print ('final count number ', c0 )
            self.peaks_detail_stack= self.peaks_detail_stack[0:c0]
            self.drop_peaks_detail_stack= self.drop_peaks_detail_stack[0:c0]
            self.iters_stack= self.iters_stack[0:c0]
            self.iters_len=c0
    
    def get_min_cost_candidate(self, peaks_target, drop_peaks_target, NRANK=10, symmetry=False, disp=False, grid_peaks=False):
        #
        cost_list=np.zeros(self.iters_len)
        
        if self.USE_COST_RATIO:
            peaks_target_ratio= peaks_target/peaks_target[0]
            peaks_target_ratio[0]=peaks_target[0]
            peaks_target2=peaks_target_ratio
            if drop_peaks_target is not None:
                drop_peaks_target_ratio= drop_peaks_target/drop_peaks_target[0]
                drop_peaks_target_ratio[0]=drop_peaks_target[0]
                drop_peaks_target2=drop_peaks_target_ratio
            else:
                drop_peaks_target2 =None
        else:
            peaks_target2=peaks_target
            drop_peaks_target2=drop_peaks_target
        
        for i in range( self.iters_len ):
            cost_list[i]= self.tube.cost_0(self.peaks_detail_stack[i], self.drop_peaks_detail_stack[i],\
             peaks_target2, drop_peaks_target2, USE_COST_RATIO=self.USE_COST_RATIO)
        
        # sort, [0] is minimum candidate
        self.rank_index= np.argsort( cost_list )
        self.rank_value= np.sort( cost_list )
        
        if disp:
            # show Top NRANK value...
            for i in range (NRANK):
                print (self.rank_index[i], self.rank_value[i], self.iters_stack[ self.rank_index[i]])
        
        # select based on shape symmetry
        select_index=0
        if symmetry:
            if self.iters_stack.shape[1] == 3:  # two tube
                cost00= self.rank_value[0]
                for i in range (3): # search top 3
                    cost0i= self.rank_value[i]
                    L1= self.iters_stack[self.rank_index[i]][0]
                    L2= self.iters_stack[self.rank_index[i]][1]
                    if L1 >= L2 and abs( cost0i - cost00) < 1.0 : # get  L1 >= L2 and cost0 difference < 1.0( 1.0 is tentative value)
                        select_index=i
                        break
                #print (' select_index', select_index)
            elif self.iters_stack.shape[1] == 5:  # three tube
                # not implemented yet
                pass
        if self.USE_COST_RATIO:
            z= self.iters_stack[ self.rank_index[select_index]].copy()
            ratio0= self.peaks_detail_stack[self.rank_index[select_index]][0] / peaks_target[0]
            
            if self.iters_stack.shape[1] == 3:  # two tube
                z[0]=z[0] * ratio0  # modified L1
                z[1]=z[1] * ratio0  # modified L2
            elif self.iters_stack.shape[1] == 5:  # three tube
                z[0]=z[0] * ratio0  # modified L1
                z[1]=z[1] * ratio0  # modified L2
                z[2]=z[2] * ratio0  # modified L3
            elif self.iters_stack.shape[1] == 7:  # four tube
                z[0]=z[0] * ratio0  # modified L1
                z[1]=z[1] * ratio0  # modified L2
                z[2]=z[2] * ratio0  # modified L3
                z[3]=z[3] * ratio0  # modified L4
            elif self.iters_stack.shape[1] == 9:  # five tube
                z[0]=z[0] * ratio0  # modified L1
                z[1]=z[1] * ratio0  # modified L2
                z[2]=z[2] * ratio0  # modified L3
                z[3]=z[3] * ratio0  # modified L4
                z[4]=z[4] * ratio0  # modified L5
            if grid_peaks:
                y= self.peaks_detail_stack[self.rank_index[select_index]].copy()
                y[0]=y[0] / ratio0
                y[1:]=y[1:] * y[0]
                return z,y
            else:
                return z
        else:
            if grid_peaks:
                return self.iters_stack[ self.rank_index[select_index]].copy(), self.peaks_detail_stack[self.rank_index[select_index]].copy()
            else:
                return self.iters_stack[ self.rank_index[select_index]].copy()

if __name__ == '__main__':
    #
    from tube_peak5 import *
    
    parser = argparse.ArgumentParser(description='make precomputed data to estimate tube model')
    parser.add_argument('--tube',  '-t', type=int, default=5, help='specify number of tube, 5')
    args = parser.parse_args()
    
    if args.tube == 5:  # try five tube model
        NUM_TUBE=5
        
        Whole_tube_length= 12.  # specify  whole tube length. This version is using frequency ratio.
        #LA_ranges=(slice(1.0,Whole_tube_length,2.0),slice(1.0,Whole_tube_length,2.0),slice(1.0,Whole_tube_length,2.0),slice(1.0,Whole_tube_length,2.0),slice(1.0,Whole_tube_length,2.0),slice(-0.9, 0.9, 0.3),slice(-0.9, 0.9, 0.3),slice(-0.9, 0.9, 0.3),slice(-0.9, 0.9, 0.3)) 
        
        # ↓この粒度で計算するのに、約６時間かかった。出力サイズは約85MB
        LA_ranges=(slice(1.0,Whole_tube_length,1.0),slice(1.0,Whole_tube_length,1.0),slice(1.0,Whole_tube_length,1.0),slice(1.0,Whole_tube_length,1.0),slice(1.0,Whole_tube_length,1.0),slice(-0.9, 0.9, 0.15),slice(-0.9, 0.9, 0.15),slice(-0.9, 0.9, 0.15),slice(-0.9, 0.9, 0.15)) 
        
        #LA_ranges=(slice(0.5,Whole_tube_length,0.5),slice(0.5,Whole_tube_length,0.5),slice(0.5,Whole_tube_length,0.5),slice(0.5,Whole_tube_length,0.5),slice(0.5,Whole_tube_length,0.5),slice(-0.9, 0.9, 0.1),slice(-0.9, 0.9, 0.1),slice(-0.9, 0.9, 0.1),slice(-0.9, 0.9, 0.1)) 
    elif args.tube == 4:  # try four tube model
        NUM_TUBE=4
        
        Whole_tube_length= 10.  # specify  whole tube length. This version is using frequency ratio.
        #LA_ranges=(slice(1.0,Whole_tube_length,1.0),slice(1.0,Whole_tube_length,1.0),slice(1.0,Whole_tube_length,1.0),slice(1.0,Whole_tube_length,1.0),slice(-0.9, 0.9, 0.3),slice(-0.9, 0.9, 0.3),slice(-0.9, 0.9, 0.3)) 
        LA_ranges=(slice(1.0,Whole_tube_length,1.0),slice(1.0,Whole_tube_length,1.0),slice(1.0,Whole_tube_length,1.0),slice(1.0,Whole_tube_length,1.0),slice(-0.9, 0.9, 0.15),slice(-0.9, 0.9, 0.15),slice(-0.9, 0.9, 0.15)) 
        #LA_ranges=(slice(0.5,Whole_tube_length,0.5),slice(0.5,Whole_tube_length,0.5),slice(0.5,Whole_tube_length,0.5),slice(0.5,Whole_tube_length,0.5),slice(-0.9, 0.9, 0.1),slice(-0.9, 0.9, 0.1),slice(-0.9, 0.9, 0.1)) 
    elif args.tube == 3:  # try three tube model
        NUM_TUBE=3
        
        Whole_tube_length= 10.  # specify  whole tube length. This version is using frequency ratio.
        LA_ranges=(slice(0.5,Whole_tube_length,0.5),slice(0.5,Whole_tube_length,0.5),slice(0.5,Whole_tube_length,0.5),slice(-0.9, 0.9, 0.1),slice(-0.9, 0.9, 0.1)) 
    elif args.tube == 2:  # two tube model
        NUM_TUBE=2
        
        Whole_tube_length= 10.  # specify  whole tube length. This version is using frequency ratio.
        LA_ranges=(slice(0.5,Whole_tube_length,0.5),slice(0.5,Whole_tube_length,0.5),slice(-0.9, 0.9, 0.1))
    
    # instance
    tube= compute_tube_peak(NUM_TUBE=NUM_TUBE, rough_search=True, skip_mode=True)  #, disp=True)
    
    path0= 'pks_dpks_stack_tube_use_ratio' + str(tube.NUM_TUBE) + '.npz'
    # new pre-compute
    pc0= pre_comute(tube, LA_ranges, Whole_tube_length, path0=path0)
    
    # test to load pre-computed gird data
    #pc1= pre_comute(tube, path0=path0)
    
