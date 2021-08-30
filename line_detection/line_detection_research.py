### Read Me (an apology)
"""
this file is a reaserch file, it documents all the different line detection algorithems I played around with,
although the different versions of line detection are based one on the other, it suffers from heavy code duplication,
in this file I excercise the unholy tredition of copy-pasting code over and over again, because of the experimental and documnting nature of this document.
I commit such a hanies crime here, because I attain a strict rule in this document:
after a function in this file has been analised and documented in the workflow html file, it CAN NOT be aultered.
this rule is perpendicular to the accepted way of designing program code, (in wich the code of a program should and will change many times).
so, because of the experamential nature of this work and of the way I document my work (I document it for me, so I will remember what I had done)
I have no choice but to copy past code from one version to the next.
"""

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

def score_pixel(color, line_detector_score, position, image_size):
    y_rel = position[1]/image_size[1]
    x_rel_centered = position[0]/image_size[0] - 0.5
    if y_rel <= 0.5:
        return 0
    pos_score = np.exp(-x_rel_centered**2)
    brightness = (color[0] + color[1] + color[2])/3 # between 0 and 1
    red_shift = 0.05
    std_of_hue_diff = 0.06
    hue_score = np.exp(-(
                         (color[0]-(brightness+red_shift*2/3))**2+
                         (color[1]-(brightness-red_shift/3))**2+
                         (color[2]-(brightness-red_shift/3))**2
                        )/(2*std_of_hue_diff**2))
    brightness_score = 1-brightness
    return pos_score*(hue_score**2)*brightness_score*line_detector_score

def line_detector_v1(rgb_arr,neighbors_radius):
    # todo: this should be -y''/(y'^2), because of the symmetry of the strings interference.
    # y'^2 = (dr/dx)^2 + (dg/dx)^2 + (db/dx)^2 + (dr/dy)^2 + (dg/dy)^2 + (db/dy)^2
    # y'' = (d/dx)^2[r] + (d/dx)^2[g] + (d/dx)^2[b] + (d/dy)^2[r] + (d/dy)^2[g] + (d/dy)^2[b]
    # these derivatives_calculations should have a calculation radius of like ~8.
    smeared = np.full(rgb_arr.shape,np.nan)
    smeared[:, :, 0] = gaussian_filter(rgb_arr[:,:,0], sigma=neighbors_radius)
    smeared[:, :, 1] = gaussian_filter(rgb_arr[:,:,1], sigma=neighbors_radius)
    smeared[:, :, 2] = gaussian_filter(rgb_arr[:,:,2], sigma=neighbors_radius)
    smeared[:, :, 3] = rgb_arr[:,:,3]
    return np.linalg.norm(rgb_arr-smeared,axis=2)

def line_detector_v2(rgb_arr,first_derivative_radius,second_derivative_radius):
    # y'^2 = (dr/dx)^2 + (dg/dx)^2 + (db/dx)^2 + (dr/dy)^2 + (dg/dy)^2 + (db/dy)^2
    # y'' = (d/dx)^2[r] + (d/dx)^2[g] + (d/dx)^2[b] + (d/dy)^2[r] + (d/dy)^2[g] + (d/dy)^2[b]
    # these derivatives_calculations should have a calculation radius of like ~8.

    # first derivative
    derivative_kernel_1d = np.zeros(first_derivative_radius*2+1)
    derivative_kernel_1d[-1] = 1/(first_derivative_radius*2)
    derivative_kernel_1d[0] = -derivative_kernel_1d[-1]
    x,y = np.meshgrid(*([derivative_kernel_1d]*2))
    derivative_kernel = x + 1j*y
    derivative = np.full((*rgb_arr.shape[:2],3),0 + 0j)
    for i in range(3):
        derivative[:,:,i] = convolve2d(rgb_arr[:,:,i],derivative_kernel,mode="same",boundary="wrap")
    derivative_magnitude = np.sum(np.absolute(derivative)**2,-1)

    # second derivative
    derivative2_kernel_1d = np.zeros(second_derivative_radius*2+1)
    derivative2_kernel_1d[0] = derivative2_kernel_1d[-1] = 1 / (first_derivative_radius ** 2)
    derivative2_kernel_1d[second_derivative_radius] = -2*derivative2_kernel_1d[0]
    x,y = np.meshgrid(*([derivative2_kernel_1d]*2))
    derivative2_kernel = x + 1j*y
    second_derivative = np.full((*rgb_arr.shape[:2],3),0 + 0j)
    for i in range(3):
        second_derivative[:,:,i] = convolve2d(rgb_arr[:,:,i],derivative2_kernel,mode="same",boundary="wrap")
    second_derivative_magnitude = np.sum(np.real(second_derivative) + np.imag(second_derivative),-1)

    # combine derivatives
    nonatomic_derivative_magnitude = np.maximum(derivative_magnitude,0.03)
    return second_derivative_magnitude/nonatomic_derivative_magnitude

def line_detector_v3(rgb_arr,void_width,line_width):
    # line fit kernel
    lfk = np.zeros(void_width + 1)
    lfk[void_width//2 - line_width//2:void_width//2 + line_width//2 + 1] = -1

    lfk -= np.mean(lfk)
    x,y = np.meshgrid(*([lfk]*2))
    line_fit_kernel = x + 1j*y
    fit_value = np.full((*rgb_arr.shape[:2],3),0 + 0j)
    for i in range(3):
        fit_value[:,:,i] = convolve2d(rgb_arr[:,:,i],line_fit_kernel,mode="same",boundary="wrap")
    fit_value_magnitude = np.sum(np.real(fit_value) + np.imag(fit_value),-1)
    return fit_value_magnitude

def line_detector_v3p1(rgb_arr,line_width,void_width=None):
    if void_width == None:
        void_width = line_width * 2
    # line fit kernel
    lfk = np.zeros(void_width + 1)
    lfk[void_width//2 - line_width//2:void_width//2 + line_width//2 + 1] = -1

    lfk -= np.mean(lfk)
    x,y = np.meshgrid(*([lfk]*2))
    line_fit_kernel = x + 1j*y
    fit_value = np.full((*rgb_arr.shape[:2],3),0 + 0j)
    for i in range(3):
        fit_value[:,:,i] = convolve2d(rgb_arr[:,:,i],line_fit_kernel,mode="same",boundary="wrap")
    fit_value_magnitude = np.sum(np.maximum(np.real(fit_value), np.imag(fit_value)),-1)
    return fit_value_magnitude

def line_detector_v3p2(rgb_arr,line_width):
    fit_value = np.full((*rgb_arr.shape[:2],3),0.0 + 0.0j)
    for i in range(3):
        fit_value[:,:,i] = gaussian_filter(rgb_arr[:, :, i], line_width/2, order=[0, 2], mode="constant", cval=0, truncate=4,output=float) +\
                           gaussian_filter(rgb_arr[:, :, i], line_width/2, order=[2, 0], mode="constant", cval=0, truncate=4,output=float)*1j
    fit_value_magnitude = np.sum(np.maximum(np.real(fit_value), np.imag(fit_value)),-1)
    return fit_value_magnitude

def line_detector_v3p3(rgb_arr,line_width):

    # line fit kernel
    if line_width != 1:
        void_width = line_width * 2
        lfk = np.zeros(void_width)
        if line_width % 2 == 0:
            lfk[void_width // 2 - line_width // 2:void_width // 2 + line_width // 2] = -1
        else:
            lfk[void_width // 2 - line_width // 2:void_width // 2 + line_width // 2] = -1
            lfk[line_width // 2] = -0.5
            lfk[-line_width // 2] = -0.5
    else:
        lfk = [0, -1, 0]

    x = np.repeat(lfk,len(lfk)).reshape((len(lfk),len(lfk)))
    diag1 = rotate(x,45,reshape=False,mode="nearest",order=1)
    diag2 = rotate(x, -45, reshape=False,mode="nearest",order=1)
    y = rotate(x, 90, reshape=False)

    diag2 -= np.mean(diag2)
    diag1 -= np.mean(diag1)
    x -= np.mean(x)
    y -= np.mean(y)

    fit_value = np.zeros((*rgb_arr.shape[:2],3))
    for i in range(3):
        fit_value[:,:,i] = np.maximum.reduce([
            convolve2d(rgb_arr[:, :, i], x, mode="same", boundary="wrap"),
            convolve2d(rgb_arr[:, :, i], diag1, mode="same", boundary="wrap"),
            convolve2d(rgb_arr[:, :, i], diag2, mode="same", boundary="wrap"),
            convolve2d(rgb_arr[:, :, i], y, mode="same", boundary="wrap")
        ])
    fit_value_magnitude = np.sum(fit_value**2,-1)
    return fit_value_magnitude

def line_detector_v3p4(rgb_arr,line_width):

    # line fit kernel
    if line_width != 1:
        void_width = line_width * 2
        lfk = np.zeros(void_width)
        if line_width % 2 == 0:
            lfk[void_width // 2 - line_width // 2:void_width // 2 + line_width // 2] = -1
        else:
            lfk[void_width // 2 - line_width // 2:void_width // 2 + line_width // 2] = -1
            lfk[line_width // 2] = -0.5
            lfk[-line_width // 2] = -0.5
    else:
        lfk = [0, -1, 0]

    x_ker = np.repeat(lfk,len(lfk)).reshape((len(lfk),len(lfk)))
    diag1_ker = rotate(x_ker,-45,reshape=False,mode="nearest",order=1)
    diag2_ker = None # will be defined as the mirror of diag1_ker when diag1_ker will be finished
    y_ker = rotate(x_ker, 90, reshape=False)

    x_ker -= np.mean(x_ker)
    y_ker -= np.mean(y_ker)

    # normalise orthogonal diagonals of diagonal kernels
    for x_intersection in range(diag1_ker.shape[0] + diag1_ker.shape[1] - 1):
        x_indexes = np.arange(x_intersection+1)
        y_indexes = x_intersection - x_indexes
        indexes_that_are_in_bounds = (x_indexes < diag1_ker.shape[0])*(y_indexes < diag1_ker.shape[1])
        x_indexes = x_indexes[indexes_that_are_in_bounds]
        y_indexes = y_indexes[indexes_that_are_in_bounds]
        diag1_ker[x_indexes,y_indexes] -= np.mean(diag1_ker[x_indexes,y_indexes])

    diag2_ker = np.flip(diag1_ker,0)

    fit_value = np.zeros((*rgb_arr.shape[:2],3))
    for i in range(3):
        fit_value[:,:,i] = np.maximum.reduce([
            convolve2d(rgb_arr[:, :, i], x_ker, mode="same", boundary="wrap"),
            convolve2d(rgb_arr[:, :, i], diag1_ker, mode="same", boundary="wrap"),
            convolve2d(rgb_arr[:, :, i], diag2_ker, mode="same", boundary="wrap"),
            convolve2d(rgb_arr[:, :, i], y_ker, mode="same", boundary="wrap")
        ])
    fit_value_magnitude = np.sum(fit_value**2,-1)
    return fit_value_magnitude

def line_detector_v3p5(rgb_arr,line_width):

    # line fit kernel
    if line_width != 1:
        void_width = line_width * 2
        lfk = np.zeros(void_width)
        if line_width % 2 == 0:
            lfk[void_width // 2 - line_width // 2:void_width // 2 + line_width // 2] = -1
        else:
            lfk[void_width // 2 - line_width // 2:void_width // 2 + line_width // 2] = -1
            lfk[line_width // 2] = -0.5
            lfk[-line_width // 2] = -0.5
    else:
        lfk = [0, -1, 0]

    x_ker = np.repeat(lfk,len(lfk)).reshape((len(lfk),len(lfk)))
    diag1_ker = rotate(x_ker,-45,reshape=True,order=1)
    diag2_ker = None # will be defined as the mirror of diag1_ker when diag1_ker will be finished
    y_ker = np.swapaxes(x_ker,0,1)

    # normalise diag1_kernel
    rot_mask = rotate(x_ker*0+1,-45,reshape=True,order=1)
    diag1_ker -= np.sum(diag1_ker)/np.sum(rot_mask)
    diag1_ker *= rot_mask

    x_ker -= np.mean(x_ker)
    y_ker -= np.mean(y_ker)
    diag2_ker = np.flip(diag1_ker,0)

    fit_value = np.zeros((*rgb_arr.shape[:2],3))
    for i in range(3):
        fit_value[:,:,i] = np.maximum.reduce([
            convolve2d(rgb_arr[:, :, i], x_ker, mode="same", boundary="wrap"),
            convolve2d(rgb_arr[:, :, i], diag1_ker, mode="same", boundary="wrap"),
            convolve2d(rgb_arr[:, :, i], diag2_ker, mode="same", boundary="wrap"),
            convolve2d(rgb_arr[:, :, i], y_ker, mode="same", boundary="wrap")
        ])
    fit_value_magnitude = np.sum(fit_value**2,-1)
    return fit_value_magnitude

def line_detector_v3p6(rgb_arr,line_width):

    # line fit kernel
    if line_width != 1:
        void_width = line_width * 2
        lfk = np.zeros(void_width)
        if line_width % 2 == 0:
            lfk[void_width // 2 - line_width // 2:void_width // 2 + line_width // 2] = -1
        else:
            lfk[void_width // 2 - line_width // 2:void_width // 2 + line_width // 2] = -1
            lfk[line_width // 2] = -0.5
            lfk[-line_width // 2] = -0.5
    else:
        lfk = [0, -1, 0]

    x_ker = np.repeat(lfk,len(lfk)).reshape((len(lfk),len(lfk)))
    diag1_ker = rotate(x_ker,-45,reshape=True,order=1)
    diag2_ker = None # will be defined as the mirror of diag1_ker when diag1_ker will be finished
    y_ker = np.swapaxes(x_ker,0,1)

    # normalise diag1_kernel
    rot_mask = rotate(x_ker*0+1,-45,reshape=True,order=1)
    diag1_ker -= np.sum(diag1_ker)/np.sum(rot_mask)
    diag1_ker *= rot_mask

    x_ker -= np.mean(x_ker)
    y_ker -= np.mean(y_ker)
    diag2_ker = np.flip(diag1_ker,0)

    fit_value = np.zeros((*rgb_arr.shape[:2],3))
    for i in range(3):
        fit_value[:,:,i] = np.maximum.reduce([
            convolve2d(rgb_arr[:, :, i], x_ker, mode="same", boundary="symm"),
            convolve2d(rgb_arr[:, :, i], diag1_ker, mode="same", boundary="symm"),
            convolve2d(rgb_arr[:, :, i], diag2_ker, mode="same", boundary="symm"),
            convolve2d(rgb_arr[:, :, i], y_ker, mode="same", boundary="symm")
        ])
    fit_value_magnitude = np.sum(fit_value,-1)
    return fit_value_magnitude

def init_detector_v4(line_width,std_pixels=100):
    if line_width != 4:
        raise NotImplementedError
    global line_detector_v4_angular_dict
    line_detector_v4_angular_dict = {}
    num_of_angles = 500
    num_of_fit_val_points = 200

    angles = np.linspace(0, 180, num_of_angles)
    fit_vals = np.linspace(0, 0.675, num_of_fit_val_points)
    y_sns = np.maximum(np.exp(-(angles - 90) ** 2 / (2 * 29.5 ** 2)) * 0.69 - 0.03, 0)
    x_sns = np.interp((angles + 90) % 180, angles, y_sns)
    d_centered_sns = np.maximum(np.exp(-(angles - 90) ** 2 / (2 * 28 ** 2)) * 0.681 - 0.021, 0)
    d1_sns = np.interp((angles + 45) % 180, angles, d_centered_sns)
    d2_sns = np.interp((angles - 45) % 180, angles, d_centered_sns)

    # generate matrix form
    def gen_matrix_form_v2(function_form, std_pixels):
        matrix_from = np.zeros((num_of_angles, num_of_fit_val_points))
        for fit_val_index in range(len(fit_vals)):
            fit_val = fit_vals[fit_val_index]
            diff_vect = np.abs(function_form - fit_val)
            matrix_from[:, fit_val_index] = diff_vect
        matrix_from = np.exp(-matrix_from * std_pixels)
        matrix_from /= np.repeat(np.sum(matrix_from, 0)[np.newaxis, :], num_of_angles, 0)
        return matrix_from

    x_mat = gen_matrix_form_v2(x_sns, std_pixels)
    y_mat = gen_matrix_form_v2(y_sns, std_pixels)
    d1_mat = gen_matrix_form_v2(d1_sns, std_pixels)
    d2_mat = gen_matrix_form_v2(d2_sns, std_pixels)
    matrices = np.array([x_mat,d1_mat,y_mat,d2_mat])

    def get_fit_val_indexes(measured):
        f_index = np.interp(measured, fit_vals, range(num_of_fit_val_points),left=0,right=num_of_fit_val_points-1.001)
        indx_bot = f_index.astype(int)
        part_in_index_bot = 1 - f_index % 1
        indx_top = indx_bot + 1
        part_in_index_top = f_index % 1
        return [indx_bot,part_in_index_bot,indx_top,part_in_index_top]

    def get_cyclic_mean_and_std(values,cycle_size,weights=None,axis=0):
        if weights is None:
            weights = np.full(values.shape,1/values.shape[axis])
        mean_complex = np.sum(weights * np.exp(1j * values / cycle_size * 2 * np.pi),axis)
        if len(values.shape) != 1:
            mean_complex_extanded = np.stack([mean_complex]*values.shape[axis],axis=axis)
        var = np.sum(weights * np.absolute(np.exp(1j * values / cycle_size * 2 * np.pi) - mean_complex_extanded)**2,axis=axis)
        return np.angle(mean_complex) * cycle_size / (2 * np.pi), np.sqrt(var)

    def get_mean_angle(fit_x,fit_y,fit_diag1,fit_diag2,b_skip_array):
        fit_diag1 = np.array(fit_diag1) * 1.2
        fit_diag2 = np.array(fit_diag2) * 1.2
        sum_of_vals = fit_x + fit_y + fit_diag1 + fit_diag2
        normalised_fit_measurments = np.array([fit_x,fit_diag1,fit_y,fit_diag2]) / sum_of_vals
        normalised_fit_measurments_indexes_bot,normalised_fit_measurments_part_bot,\
        normalised_fit_measurments_indexes_top,normalised_fit_measurments_part_top = get_fit_val_indexes(normalised_fit_measurments)
        mean_angle = np.full(fit_x.shape,np.nan,float)
        std = np.full(fit_x.shape,np.nan,float)
        for pix_x_index,pix_y_index in pbar_itertools.product(range(mean_angle.shape[0]),range(mean_angle.shape[1])):
            if not b_skip_array[pix_x_index,pix_y_index]:
                bot_index = normalised_fit_measurments_indexes_bot[:,pix_x_index,pix_y_index]
                bot_part = normalised_fit_measurments_part_bot[:,pix_x_index,pix_y_index]
                top_index = normalised_fit_measurments_indexes_top[:,pix_x_index,pix_y_index]
                top_part = normalised_fit_measurments_part_top[:,pix_x_index,pix_y_index]
                angle_prob_arr = np.product([matrices[i,:,bot_index[i]]*bot_part[i] + matrices[i,:,top_index[i]]*top_part[i] for i in range(len(bot_index))],0)
                angle_prob_arr /= np.sum(angle_prob_arr)
                mean_angle[pix_x_index,pix_y_index], std[pix_x_index,pix_y_index] = get_cyclic_mean_and_std(angles,180,angle_prob_arr)
        return mean_angle,std

    line_detector_v4_angular_dict["num_of_angles"] = num_of_angles
    line_detector_v4_angular_dict["num_of_fit_val_points"] = num_of_fit_val_points
    line_detector_v4_angular_dict["angles"] = angles
    line_detector_v4_angular_dict["fit_vals"] = fit_vals
    line_detector_v4_angular_dict["x_mat"] = x_mat
    line_detector_v4_angular_dict["y_mat"] = y_mat
    line_detector_v4_angular_dict["d1_mat"] = d1_mat
    line_detector_v4_angular_dict["d2_mat"] = d2_mat
    line_detector_v4_angular_dict["get_mean_angle"] = get_mean_angle
    line_detector_v4_angular_dict["get_cyclic_mean_and_std"] = get_cyclic_mean_and_std

def line_detector_v4(rgb_arr,line_width):

    #load from init:
    get_mean_angle = line_detector_v4_angular_dict["get_mean_angle"]
    get_cyclic_mean_and_std = line_detector_v4_angular_dict["get_cyclic_mean_and_std"]
    # line fit kernel
    if line_width != 1:
        void_width = line_width * 2
        lfk = np.zeros(void_width)
        if line_width % 2 == 0:
            lfk[void_width // 2 - line_width // 2:void_width // 2 + line_width // 2] = -1
        else:
            lfk[void_width // 2 - line_width // 2:void_width // 2 + line_width // 2] = -1
            lfk[line_width // 2] = -0.5
            lfk[-line_width // 2] = -0.5
    else:
        lfk = [0, -1, 0]

    x_ker = np.repeat(lfk,len(lfk)).reshape((len(lfk),len(lfk)))
    diag1_ker = rotate(x_ker,-45,reshape=True,order=1)
    diag2_ker = None # will be defined as the mirror of diag1_ker when diag1_ker will be finished
    y_ker = np.swapaxes(x_ker,0,1)

    # normalise diag1_kernel
    rot_mask = rotate(x_ker*0+1,-45,reshape=True,order=1)
    diag1_ker -= np.sum(diag1_ker)/np.sum(rot_mask)
    diag1_ker *= rot_mask

    x_ker -= np.mean(x_ker)
    y_ker -= np.mean(y_ker)
    diag2_ker = np.flip(diag1_ker,0)

    fit_value = np.zeros((*rgb_arr.shape[:2],3))
    angle_value = np.zeros((*rgb_arr.shape[:2],3))

    for i in range(3):
        fit_x =convolve2d(rgb_arr[:, :, i], x_ker, mode="same", boundary="symm")
        fit_diag1 =convolve2d(rgb_arr[:, :, i], diag1_ker, mode="same", boundary="symm")
        fit_y =convolve2d(rgb_arr[:, :, i], y_ker, mode="same", boundary="symm")
        fit_diag2 =convolve2d(rgb_arr[:, :, i], diag2_ker, mode="same", boundary="symm")

        # calculate fit_value
        fit_value[:, :, i] = np.maximum.reduce([fit_x, fit_y, fit_diag1, fit_diag2])

        # calculate angle
        b_angle_skip_array = fit_value[:, :, i] < np.quantile(fit_value[:, :, i].flatten(),0.90)
        angle_value[:,:,i],angle_std = get_mean_angle(fit_x,fit_y,fit_diag1,fit_diag2,b_angle_skip_array)
    fit_value_total = np.sum(fit_value,-1)
    fit_value_error = np.std(fit_value, -1)
    angle_arr_total,angle_arr_error = get_cyclic_mean_and_std(angle_value,180,axis=-1)
    angle_arr_total *= np.pi/180
    return fit_value_total,fit_value_error,angle_arr_total,angle_arr_error

def line_detector_v3p7(rgb_arr,line_width):
    # line fit kernel
    if line_width != 1:
        void_width = line_width * 2
        lfk = np.zeros(void_width)
        if line_width % 2 == 0:
            lfk[void_width // 2 - line_width // 2:void_width // 2 + line_width // 2] = -1
        else:
            lfk[void_width // 2 - line_width // 2:void_width // 2 + line_width // 2] = -1
            lfk[line_width // 2] = -0.5
            lfk[-line_width // 2] = -0.5
    else:
        lfk = [0, -1, 0]

    x_ker = np.repeat(lfk,len(lfk)).reshape((len(lfk),len(lfk)))
    diag1_ker = rotate(x_ker,-45,reshape=True,order=1)
    diag2_ker = None # will be defined as the mirror of diag1_ker when diag1_ker will be finished
    y_ker = np.swapaxes(x_ker,0,1)

    # normalise diag1_kernel
    rot_mask = rotate(x_ker*0+1,-45,reshape=True,order=1)
    diag1_ker -= np.sum(diag1_ker)/np.sum(rot_mask)
    diag1_ker *= rot_mask

    x_ker -= np.mean(x_ker)
    y_ker -= np.mean(y_ker)
    diag2_ker = np.flip(diag1_ker,0)

    fit_value = np.zeros((*rgb_arr.shape[:2],3))
    for i in range(3):
        fit_value[:,:,i] = np.maximum.reduce([
            convolve2d(rgb_arr[:, :, i], x_ker, mode="same", boundary="symm"),
            convolve2d(rgb_arr[:, :, i], diag1_ker, mode="same", boundary="symm"),
            convolve2d(rgb_arr[:, :, i], diag2_ker, mode="same", boundary="symm"),
            convolve2d(rgb_arr[:, :, i], y_ker, mode="same", boundary="symm")
        ])
    fit_value_magnitude = np.sum(fit_value,-1)
    fit_value_std = np.std(fit_value, -1)
    return fit_value_magnitude,fit_value_std

def score_image(rgb_arr,line_width=5):
    line_detector_score = line_detector_v3p7(rgb_arr,line_width)[0]
    score_arr = np.full(rgb_arr.shape[:2],np.nan)
    for i,(y,x) in enumerate(pbar_itertools.product(range(rgb_arr.shape[0]),range(rgb_arr.shape[1]))):
        score_arr[y,x] = score_pixel(rgb_arr[y,x],line_detector_score[y,x], (x,y), rgb_arr.shape)
    return score_arr

def score_pixel_v3p2(rgb_arr,line_width=5):
    line_detector_score,line_detector_score_error = line_detector_v3p7(rgb_arr,line_width)
    score_arr = np.full(rgb_arr.shape[:2],np.nan)
    for i,(y,x) in enumerate(pbar_itertools.product(range(rgb_arr.shape[0]),range(rgb_arr.shape[1]))):
        score_arr[y,x] = score_pixel(rgb_arr[y,x],line_detector_score[y,x], (x,y), rgb_arr.shape)
    return score_arr

def inflate_bool_arr(A,inflation_radias:int):
    x,y = np.meshgrid(np.arange(inflation_radias*2 + 4,dtype=float),np.arange(inflation_radias*2 + 4,dtype=float))
    x -= np.mean(x)
    y -= np.mean(y)
    inflation_kernel = (x**2 + y**2 < (inflation_radias**2))
    inflation_kernel = inflation_kernel.astype(int)
    return convolve2d(A,inflation_kernel, mode="same", boundary="fill",fillvalue=False) > 0


if __name__ == "_a_main__":
    im = plt.imread(r"..\cropped photos\01_mw552.9_ms2.48_mp0.525.png")

    init_detector_v4(4,std_pixels=500)
    fit_value_total,fit_value_error,angle_arr_total,angle_arr_error = line_detector_v4(im,4)
    angle_arr_rel_error = 1 - angle_arr_error / np.max(angle_arr_error)

    plt.figure()
    skip = 4
    x, y = np.meshgrid(np.arange(angle_arr_total.shape[1]), np.arange(angle_arr_total.shape[0]))
    angle_arr_total_avr = np.full((angle_arr_total.shape[0] // skip + 1, angle_arr_total.shape[1] // skip + 1), np.nan)
    angle_arr_total_error = np.full((angle_arr_total.shape[0] // skip + 1, angle_arr_total.shape[1] // skip + 1), np.nan)
    for xi, yi in pbar_itertools.product(np.arange(0, angle_arr_total_avr.shape[1]), np.arange(0, angle_arr_total_avr.shape[0])):
        angle_arr_total_avr[yi, xi] = np.angle(
            np.nanmean(np.exp(1j * angle_arr_total[yi * skip - skip // 2:yi * skip + skip // 2, xi * skip - skip // 2:xi * skip + skip // 2])))
        angle_arr_total_error = np.nanmean(angle_arr_rel_error[yi * skip - skip // 2:yi * skip + skip // 2, xi * skip - skip // 2:xi * skip + skip // 2])
    plt.quiver(x[::skip, ::skip], y[::skip, ::skip], skip*np.cos(angle_arr_total_avr), skip*np.sin(angle_arr_total_avr), angles='xy', units='xy',
               scale_units='xy', scale=1, width=skip/4, facecolor="b", edgecolor=(1, 1, 1),alpha=angle_arr_total_error,
               linewidths=0.5, headlength=2, headwidth=2, headaxislength=2)
    plt.imshow(fit_value_total, vmin=0, origin="upper", alpha=0.4, cmap="gray")

    # lds4 = line_detector_v3p6(im, 4)
    # lds7 = line_detector_v3p6(im, 7)
    #
    # plt.figure()
    # plt.suptitle("line detector v3.6")
    # plt.subplot(121)
    # plt.title("line_width = 4")
    # plt.imshow(lds4,vmin=0)
    # plt.subplot(122)
    # plt.title("line_width = 7")
    # plt.imshow(lds7, vmin=0)

    # string_tight = np.load("string_location_points.npy")
    #
    # plt.figure()
    # score_arr = {lw:score_image(im, lw) for lw in range(2,8)}
    # x = 3
    # y = 2
    # for i,lw in enumerate(score_arr.keys()):
    #     plt.subplot(y,x,i+1)
    #     plt.title("total pixel score v3, with line_width = {}".format(lw))
    #     plt.imshow(score_arr[lw],vmin=0)
    #
    # plt.figure()
    # plt.suptitle("density histograms of SNR for differant linewidths total_pixel_score_v3")
    # x = 3
    # y = 2
    # for i,lw in enumerate(score_arr.keys()):
    #     color1 = (0.2,0.4,0.7)
    #     color2 = "r"
    #     distance_from_string = lw
    #     string = inflate_bool_arr(string_tight, distance_from_string)
    #
    #     plt.subplot(y,x,i+1)
    #     plt.title("total pixel score v3, with line_width = {}".format(lw))
    #     bins = np.linspace(np.min(score_arr[lw]),np.max(score_arr[lw]),int(score_arr[lw].size**0.5))
    #     temp = np.copy(score_arr[lw][~string].flatten())
    #     noise = temp[temp != 0].flatten()
    #     signal = score_arr[lw][string].flatten()
    #     distrb_nois,_,_ = plt.hist(noise, bins, alpha=0.8, color=color1, label="distance from string > {}".format(distance_from_string),density=True)
    #     distrb_signl,_,_ = plt.hist(signal, bins, alpha=0.5, color=color2, label="distance from string < {}".format(distance_from_string),density=True)
    #
    #     maxy = np.partition(distrb_nois, -3)[-3]
    #
    #     nois_width = np.quantile(noise,0.95) - np.quantile(noise,0.05)
    #     signal_mean = np.mean(signal[signal > np.quantile(noise,0.95)])
    #
    #     plt.vlines([np.quantile(noise,0.95), np.quantile(noise,0.05)],0,maxy,colors=color1,linestyles="--",label="90% confidance interval\nof noise width\n  = {:.4f}".format(nois_width))
    #     plt.vlines(signal_mean, 0, maxy, colors=color2, linestyles="--",label="mean of signal\nthat is grater\nthan 95% of noise\n  = {:.1f}*noise_width".format(signal_mean/nois_width))
    #
    #     # plt.yscale("log")
    #     plt.ylim(0,maxy)
    #     plt.xlim(np.quantile(signal,0.01),np.quantile(signal,0.99))
    #     plt.legend(loc="upper right")
    #     plt.xlabel("score of pixel")
    #     plt.ylabel("density")
    #     plt.grid(linestyle="--")
    #
    # plt.show()
