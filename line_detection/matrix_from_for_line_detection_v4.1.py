# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d
import os
from tqdm import tqdm as pbar
from tqdm.contrib import itertools as pbar_itertools
from typing import List, Tuple
from scipy.ndimage import gaussian_filter, rotate
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
# import itertools
from scipy.interpolate import interp1d

Point = Tuple[float, float]
import addcopyfighandler

addcopyfighandler.image_file_format = "png"
addcopyfighandler.image_dpi = 200

num_of_angles = 500
num_of_fit_val_points = 200

angles = np.linspace(0, 180, num_of_angles)
fit_vals = np.linspace(0, 0.675, num_of_fit_val_points)
y_sns = np.maximum(np.exp(-(angles - 90) ** 2 / (2 * 29.5 ** 2)) * 0.69 - 0.03, 0)
x_sns = np.interp((angles + 90)%180,angles,y_sns)
d_centered_sns = np.maximum(np.exp(-(angles - 90) ** 2 / (2 * 28 ** 2)) * 0.681 - 0.021, 0)
d1_sns = np.interp((angles + 45)%180,angles,d_centered_sns)
d2_sns = np.interp((angles - 45)%180,angles,d_centered_sns)


# generate matrix form
def gen_matrix_form_v1(function_form):
    matrix_from = np.zeros((num_of_angles, num_of_fit_val_points))
    for fit_val_index in range(len(fit_vals)):
        fit_val = fit_vals[fit_val_index]
        angles_bot_indexes = np.where(np.diff(np.sign(function_form - fit_val)))
        for angle_bot_index in angles_bot_indexes:
            part_in_bot = 1 - (fit_val - function_form[angle_bot_index])/(function_form[angle_bot_index + 1] - function_form[angle_bot_index])
            matrix_from[angle_bot_index, fit_val_index] += part_in_bot
            matrix_from[angle_bot_index + 1, fit_val_index] += 1 - part_in_bot
    return matrix_from

def gen_matrix_form_v2(function_form,std_pixels):
    matrix_from = np.zeros((num_of_angles, num_of_fit_val_points))
    for fit_val_index in range(len(fit_vals)):
        fit_val = fit_vals[fit_val_index]
        diff_vect = np.abs(function_form - fit_val)
        matrix_from[:,fit_val_index] = diff_vect
    matrix_from = np.exp(-matrix_from*std_pixels)
    matrix_from /= np.repeat(np.sum(matrix_from,0)[np.newaxis,:],num_of_angles,0)
    return matrix_from



x_mat = gen_matrix_form_v2(x_sns,100)
y_mat = gen_matrix_form_v2(y_sns,100)
d1_mat = gen_matrix_form_v2(d1_sns,100)
d2_mat = gen_matrix_form_v2(d2_sns,100)


def mk_f_arr(angle,function_form):
    measured = np.interp(angle,angles,function_form)
    f_index = np.interp(measured,fit_vals,range(num_of_fit_val_points))
    f_arr = np.zeros(num_of_fit_val_points)
    f_arr[int(f_index)] = 1 - f_index % 1
    f_arr[int(f_index) + 1] = f_index % 1
    return f_arr

plt.figure()
angle_prob_arr = np.matmul(x_mat,mk_f_arr(0,x_sns))
plt.plot(angles,angle_prob_arr,label="angle probability from x_ker_mat")

angle_prob_arr = np.matmul(y_mat,mk_f_arr(0,y_sns))
plt.plot(angles,angle_prob_arr,label="angle probability from y_ker_mat")

angle_prob_arr = np.matmul(d1_mat,mk_f_arr(0,d1_sns))
plt.plot(angles,angle_prob_arr,label="angle probability from d1_ker_mat")

angle_prob_arr = np.matmul(d2_mat,mk_f_arr(0,d2_sns))
plt.plot(angles,angle_prob_arr,label="angle probability from d2_ker_mat")
plt.legend()
plt.grid(ls="--")
plt.title("angle probability of differant kernels\nthe measured fit values corrispond to angle 0°\nsigma = 100")
plt.xlabel("angle in degrees [°]")
plt.ylabel("angle probability distrebution")

angle_prob_arr =  np.matmul(x_mat, mk_f_arr(0, x_sns)) * \
                  np.matmul(y_mat, mk_f_arr(0, y_sns)) * \
                  np.matmul(d1_mat, mk_f_arr(0, d1_sns)) * \
                  np.matmul(d2_mat, mk_f_arr(0, d2_sns))
angle_prob_arr /= np.sum(angle_prob_arr)
mean_complex = np.sum(angle_prob_arr*np.exp(1j*angles/180*np.pi*2))
mean_angle = np.angle(mean_complex)/2
angle_cum_prob_arr = np.cumsum(angle_prob_arr*np.gradient(angles))
uncertainty_interval = np.interp([0.025,0.975],angle_cum_prob_arr,angles)

plt.figure()
plt.plot(angles,angle_prob_arr)
plt.ylim(*plt.ylim())
plt.vlines([mean_angle],*plt.ylim(),linestyles="--",label="distribution mean")
plt.vlines([uncertainty_interval],*plt.ylim(),colors="r",linestyles="--",label="distribution 95% confidance interval")
plt.title("the product of the angle probability of all 4 kernels\nthe measured fit values corrispond to angle 0°\nsigma = 100")
plt.xlabel("angle [°]")
plt.ylabel("total probability distribution")
plt.legend(loc="upper right")
plt.grid(linestyle="--")
