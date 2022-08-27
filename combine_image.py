#coding:utf-8

#
# combine 2,3,and 4 tube model estimation result figures
#
#  For example:
#     result_figure/2tube/*.png   result figures of pks2tube5frame.py -t 2 (use 2 tube model)
#     result_figure/3tube/*.png   result figures of pks2tube5frame.py -t 3 (use 3 tube model)
#     result_figure/4tube/*.png   result figures of pks2tube5frame.py -t 4 (use 4 tube model)
#     result_figure/234tube/*.png combine above 2,3,and 4 tube model estimation result figures

import os
import argparse
import glob
from matplotlib import pyplot as plt
import numpy as np


# check version
#  python 3.6.4 on win32
#  numpy 1.18.4
#  matplotlib 3.3.1



if __name__ == '__main__':
    #
    parser = argparse.ArgumentParser(description='combine 2,3,and 4 tube model estimation result figures ')
    parser.add_argument('--result_dir', '-r', default='result_figure', help='specify result directory')
    parser.add_argument('--frame', '-f', type=int, default=-1, help='specify start frame number, igonred if negative')
    args = parser.parse_args()
    
    #
    flist= glob.glob( args.result_dir + '/2tube' + '/*.png')
    dir_out= args.result_dir + '/tube234'
    # if combine specify start frame number 4 tube model estimation result at the bottom
    if args.frame >= 0:
        dir_out = dir_out + '_' + str(args.frame)
    
    if not os.path.isdir( dir_out  ):
        os.mkdir( dir_out )
    
    
    for f in flist:
        fbase=os.path.basename(f)
        print (fbase)
        tube2=plt.imread(f)
        tube3=plt.imread( args.result_dir + '/3tube/' + fbase)
        tube4=plt.imread( args.result_dir + '/4tube/' + fbase)
        
        ep=300
        st=tube2.shape[0] - ep
        sr=35
        
        # if combine specify start frame number 4 tube model estimation result at the bottom
        if args.frame >= 0:
            tube4s=plt.imread( args.result_dir + '/' + str(args.frame) + '/' + fbase)
            tube234a=np.vstack((tube2[:st,:,:],tube3[:st,:,:],tube4[:st,:,:],tube4s[:st,:,:]))
            tube234b=np.vstack((tube2[ep:,sr:,:],tube3[ep:,sr:,:],tube4[ep:,sr:,:],tube4s[ep:,sr:,:]))
        else:
            tube234a=np.vstack((tube2[:st,:,:],tube3[:st,:,:],tube4[:st,:,:]))
            tube234b=np.vstack((tube2[ep:,sr:,:],tube3[ep:,sr:,:],tube4[ep:,sr:,:]))
        
        tube234 =np.hstack((tube234a, tube234b))
        
        #print ('shape', tube2.shape)
        #print ('shape', tube234.shape)
        
        plt.imsave( dir_out + '/' + fbase,  tube234)
        
