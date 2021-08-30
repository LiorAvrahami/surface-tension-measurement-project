# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter,rotate
from scipy.signal import convolve2d
import os
from tqdm import tqdm as pbar
from tqdm.contrib import itertools as pbar_itertools
from typing import List,Tuple
import addcopyfighandler
# import itertools
Point = Tuple[float,float]
from line_detection_research import score_image,inflate_bool_arr
import pickle

im = plt.imread(r"..\cropped photos\01_mw552.9_ms2.48_mp0.525.png")
string_tight = np.load("string_location_points.npy")

if os.path.exists("snr_calculation_string_detection"):
    with open("snr_calculation_string_detection","rb") as f:
        score_arr = pickle.load(f)
else:
    score_arr = {lw:score_image(im, lw) for lw in [2,3,5,7,10,14,18,24]}
    with open("snr_calculation_string_detection","wb+") as f:
        pickle.dump(score_arr,f)

# x = 3
# y = 2
# plt.figure()
# for i,lw in enumerate(score_arr.keys()):
#     plt.subplot(y,x,i+1)
#     plt.title("total pixel score v3, with line_width = {}".format(lw))
#     plt.imshow(score_arr[lw],vmin=0)

# calculate_snr
snr_dtb = {}

plt.figure(figsize=(18,7))
plt.suptitle("density histograms of SNR for differant linewidths total_pixel_score_v3")
x = 4
y = 2
to_draw_hist = [2,3,5,7,10,14,18,24]
i=-1
for j,lw in enumerate(score_arr):

    distance_from_string = lw
    string = inflate_bool_arr(string_tight, distance_from_string)

    temp = np.copy(score_arr[lw][~string].flatten())
    noise = temp[temp != 0].flatten()
    signal = score_arr[lw][string].flatten()

    nois_max = np.quantile(noise,0.95)
    signal_mean = np.mean(signal[signal > nois_max])
    snr_dtb[lw] = signal_mean/nois_max

    if lw in to_draw_hist:
        i+=1
        color1 = (0.2,0.4,0.7)
        color2 = "r"
        plt.subplot(y,x,i+1)
        plt.title("total pixel score v3, with line_width = {}".format(lw))

        bins = np.linspace(np.min(score_arr[lw]), np.max(score_arr[lw]), int(score_arr[lw].size ** 0.5))
        distrb_nois,_,_ = plt.hist(noise, bins, alpha=0.8, color=color1, label="background score_values".format(distance_from_string),density=True)
        distrb_signl,_,_ = plt.hist(signal, bins, alpha=0.5, color=color2, label="string score_values".format(distance_from_string),density=True)

        maxy = np.partition(distrb_nois, -3)[-3]

        plt.vlines(nois_max,0,maxy,colors=color1,linestyles="--",label="maximum_noise".format(nois_max))
        # plt.vlines(nois_mean, 0, maxy, colors=color1, linestyles="--",label="90% confidance interval\nof noise width\n  = {:.4f}".format(nois_width))
        plt.vlines(signal_mean, 0, maxy, colors=color2, linestyles="--",label="signal mean = \n{:.1f}*max_noise".format(signal_mean/nois_max))

        # plt.yscale("log")
        plt.ylim(0,maxy)
        plt.xlim(np.quantile(signal,0.01),np.quantile(signal,0.99))
        plt.legend(loc="upper right")
        plt.xlabel("score of pixel")
        plt.ylabel("density")
        plt.grid(linestyle="--")
plt.subplots_adjust(hspace=0.3)

# draw snr graph
plt.figure()
plt.plot(list(snr_dtb.keys()),list(snr_dtb.values()),"o")
plt.xlabel("\"line_width\"")
plt.ylabel("SNR - (average signal)/(maximum noise)")
plt.title("SNR vs \"line_width\"")
plt.grid(linestyle="--")


plt.show()
