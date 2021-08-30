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
from scipy.ndimage import gaussian_filter,rotate
import matplotlib.pyplot as plt
import addcopyfighandler
from scipy.interpolate import griddata
# import itertools
Point = Tuple[float,float]
addcopyfighandler.image_file_format = "png"
addcopyfighandler.image_dpi = 200


def make_angle_image(angle,lw = 5,size = (40,40)):
    # line fit kernel
    if lw != 1:
        void_width = size[0]
        lfk = np.zeros(void_width)
        lfk[void_width // 2 - lw // 2:void_width // 2 + lw // 2] = -1
        if lw % 2 != 0:
            lfk[void_width // 2 - lw // 2 - 1] = -0.5
            lfk[void_width // 2 + lw // 2] = -0.5
    else:
        lfk = [0, -1, 0]

    out = np.repeat(lfk,size[1]).reshape((len(lfk),len(lfk)))
    return rotate(out,angle,reshape=True,order=1)

def seperated_line_detector_v3p6(val_arr,line_width):

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

    return [convolve2d(val_arr, x_ker, mode="same", boundary="symm"),
           convolve2d(val_arr, diag1_ker, mode="same", boundary="symm"),
           convolve2d(val_arr, y_ker, mode="same", boundary="symm"),
           convolve2d(val_arr, diag2_ker, mode="same", boundary="symm")]

lw = 5
angles = np.linspace(0,180,1000)
x_ker_resaults = []
d1_ker_resaults = []
y_ker_resaults = []
d2_ker_resaults = []
for angle in angles:
    res = seperated_line_detector_v3p6(make_angle_image(angle),lw)

    res[1] *= 1.2
    res[3] *= 1.2

    mid_res = [res[i][res[i].shape[0] // 2, res[i].shape[1] // 2] for i in range(4)]

    res = [r/sum(mid_res) for r in res]

    x_ker_resaults.append(mid_res[0])
    d1_ker_resaults.append(mid_res[1])
    y_ker_resaults.append(mid_res[2])
    d2_ker_resaults.append(mid_res[3])

    # if any([abs(angle - a) <= np.min(np.abs(angles - a)) for a in [0,45,90,115,135,170]]):
    #     x,y = res[0].shape[0] // 2, res[0].shape[1] // 2
    #
    #     plt.figure()
    #     plt.imshow(make_angle_image(angle))
    #     plt.scatter([x],[y],c="r")
    #     plt.title(f"image of line at angle {angle:.1f}° and width {lw}")
    #
    #     plt.figure()
    #     plt.imshow(res[0],vmax=1.0)
    #     plt.scatter([x], [y], c="r")
    #     plt.title(f"x_kernel fit function of line with angle {angle:.1f}° and width {lw}")
    #
    #     plt.figure()
    #     plt.imshow(res[1],vmax=1.0)
    #     plt.scatter([x], [y], c="r")
    #     plt.title(f"diag1_kernel fit function of line with angle {angle:.1f}° and width {lw}")

x_ker_resaults = np.array(x_ker_resaults)
d1_ker_resaults = np.array(d1_ker_resaults)
y_ker_resaults = np.array(y_ker_resaults)
d2_ker_resaults = np.array(d2_ker_resaults)

x_ker_resaults_normed = x_ker_resaults / (x_ker_resaults + d1_ker_resaults + y_ker_resaults + d2_ker_resaults)
d1_ker_resaults_normed = d1_ker_resaults / (x_ker_resaults + d1_ker_resaults + y_ker_resaults + d2_ker_resaults)
y_ker_resaults_normed = y_ker_resaults / (x_ker_resaults + d1_ker_resaults + y_ker_resaults + d2_ker_resaults)
d2_ker_resaults_normed = d2_ker_resaults / (x_ker_resaults + d1_ker_resaults + y_ker_resaults + d2_ker_resaults)

x_ker_resaults = x_ker_resaults_normed
d1_ker_resaults = d1_ker_resaults_normed
y_ker_resaults = y_ker_resaults_normed
d2_ker_resaults = d2_ker_resaults_normed

del(x_ker_resaults_normed)
del(d1_ker_resaults_normed)
del(y_ker_resaults_normed)
del(d2_ker_resaults_normed)

plt.figure()
plt.plot(angles,x_ker_resaults,label="x")
plt.plot(angles,d1_ker_resaults,label="diag1")
plt.plot(angles,y_ker_resaults,label="y")
plt.plot(angles,d2_ker_resaults,label="diag2")

plt.legend()
plt.grid(ls="--")

plt.title("normalised fit value of different kernels at different line angles.\n"
          "functions are normalised so that at every x, the sum of all is 1")
plt.xlabel("angle in degrees [°]")
plt.ylabel("normalised fit value")


def sort_by_x(x,y):
    a = np.array([x,y])
    a = a[:,a[0,:].argsort()]
    return a[0],a[1]
plt.figure()
d = d1_ker_resaults
plt.plot(*sort_by_x((angles - 90 )%180 - 90,x_ker_resaults ),label="x angle sensitivity function")
plt.plot(*sort_by_x((angles - 45 )%180 - 90,d1_ker_resaults),label="diag1 angle sensitivity function")
plt.plot(*sort_by_x((angles - 0  )%180 - 90,y_ker_resaults ),label="y angle sensitivity function")
plt.plot(*sort_by_x((angles - 135)%180 - 90,d2_ker_resaults),label="diag2 angle sensitivity function")

plt.legend()
plt.grid(ls="--")
plt.title("comparison of different kernel sensitivity function forms")
plt.xlabel("angle [°]")

# paramatritize y, daig1
    # y
y = np.interp(angles,
              np.concatenate([np.array(list(reversed(180-angles[angles > 90]))),angles[angles > 90]]),
              np.concatenate([np.array(list(reversed(y_ker_resaults[angles > 90]))),y_ker_resaults[angles > 90]]))
plt.figure()
# plt.plot(angles,y_ker_resaults,label="y_kernel_fit")
plt.plot(angles,np.maximum(np.exp(-(angles-90)**2/(2*29.5**2))*0.69 - 0.0325,0),label="max(gausian_fit minus constant,0)")
plt.plot(angles,np.sum([ai*(angles-90)**i for i,ai in enumerate(reversed([
    3.01356888e-20,  0, -4.17924684e-16, 0,-1.99699176e-12,  0,  6.16006663e-08,
    0,-3.57457392e-04, 0,  6.65134361e-01]))],axis=0),label="polyfit 10")
plt.plot(angles,y,label="reflected_y_kernel_fit")
plt.legend()
plt.grid(ls="--")
plt.title("different parametrization of angle sensitivity function")
plt.xlabel("angle [°]")


    # diag1
z_ang,z_d1 = sort_by_x((angles - 45 )%180 - 90,d1_ker_resaults)
plt.figure()
# plt.plot(angles,y_ker_resaults,label="y_kernel_fit")
plt.plot(z_ang,np.maximum(np.exp(-(z_ang)**2/(2*28**2))*0.681 - 0.021,0),label="max(gausian_fit minus constant,0)")
# plt.plot(angles,np.sum([ai*(angles-90)**i for i,ai in enumerate(reversed([
#     3.01356888e-20,  0, -4.17924684e-16, 0,-1.99699176e-12,  0,  6.16006663e-08,
#     0,-3.57457392e-04, 0,  6.65134361e-01]))],axis=0),label="polyfit 10")
plt.plot(z_ang,z_d1,label="diag")
# plt.plot(angles - 90,y,label="reflected_y_kernel_fit")
plt.legend()
plt.grid(ls="--")
plt.title("different parametrization of diag's angle sensitivity function")
plt.xlabel("angle [°]")


# gau = np.exp(-(angles-90)**2/(2*29.5**2))*0.66
# np.polyfit(angles-90,y - gau,10)