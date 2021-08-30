# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d
import os
from tqdm import tqdm as pbar
from tqdm.contrib import itertools as pbar_itertools
from typing import List,Tuple
from line_detection import *
# import itertools
Point = Tuple[float,float]


im = plt.imread(r"pattern2.bmp")

def temp_compare_for_line_detection_v3(void_width, line_width):
    plt.figure(figsize=[16.5 ,  5.4])
    plt.suptitle("line detector v3.1:\nvoid_width = {},line_width = {}, vmin=0".format(void_width, line_width))
    di = line_detector_v3p1(im, line_width)
    plt.subplot(311)
    plt.imshow(im)
    plt.subplot(312)
    plt.imshow(di, cmap="jet", vmin=0)
    plt.subplot(313)
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.plot(di[70,:])
    plt.gcf().savefig("line_detection_v3p1_on_pattern2-linewidth={:02d}".format(line_width))

# a = np.zeros((100,100,3))
# a[50,50,:] = 1
# b = line_detector_v3p2(a, 4)
# plt.figure()
# plt.imshow(a,cmap="jet")
# plt.figure()
# plt.imshow(b,cmap="jet")

for i in range(8):
    temp_compare_for_line_detection_v3(void_width=i*2, line_width=i)

# for lw in range(1,15,3):
#     if lw < 10:
#         temp_compare_for_line_detection_v3(void_width=10, line_width=lw)
#     temp_compare_for_line_detection_v3(void_width=20, line_width=lw)
#     temp_compare_for_line_detection_v3(void_width=30, line_width=lw)
plt.show()