import matplotlib.pyplot as plt
import os
from line_detection import score_pixel_v3p2
from scipy.ndimage import map_coordinates
import scipy.optimize
import numpy as np
from tqdm import tqdm as pbar
from tqdm.contrib import itertools as pbar_itertools

def get_circle_from_3_points(p1,p2,p3):
    """
    :param p1,p2,p3: complex numbers representing a point on the xy plane. (y is imaginary part)
    :return: x0, y0, R
    """
    w = p3 - p1
    w /= p2 - p1
    c = (p1 - p2) * (w - abs(w) ** 2) / 2j / w.imag - p1
    return -np.real(c),-np.imag(c),np.absolute(c + p1)

def make_initial_guesses_for_circles(line_detection_score_arr,x,y):
    sx,sy = line_detection_score_arr.shape
    line_detection_score_arr = line_detection_score_arr**3
    low_image = line_detection_score_arr[:,int(sy*0.9):]
    low_x = x[:, int(sy * 0.9):]
    x_intersection = np.sum(low_x*low_image)/np.sum(low_image)

    left_image = line_detection_score_arr[:int(sx*0.2)]
    left_y = y[:int(sx*0.2)]
    y_left_intersection = np.sum(left_y * left_image) / np.sum(left_image)

    right_image = line_detection_score_arr[int(sx*0.8):]
    right_y = y[int(sx*0.8):]
    y_right_intersection = np.sum(right_y * right_image) / np.sum(right_image)

    diag_left_mask = line_detection_score_arr*np.exp(-np.abs(x + (y-sy)))
    left_d_point = np.sum(x*diag_left_mask)/np.sum(diag_left_mask),np.sum(y*diag_left_mask)/np.sum(diag_left_mask)

    diag_right_mask = line_detection_score_arr*np.exp(-np.abs((x-sx) - (y-sy)))
    right_d_point = np.sum(x * diag_right_mask) / np.sum(diag_right_mask), np.sum(y * diag_right_mask) / np.sum(diag_right_mask)

    x0_c1, y0_c1, R_c1 = get_circle_from_3_points(x_intersection + sy * 1j,
                                                  y_left_intersection * 1j,
                                                  left_d_point[0] + left_d_point[1] * 1j)

    x0_c2, y0_c2, R_c2 = get_circle_from_3_points(x_intersection + sy*1j,
                                                  sx + y_right_intersection*1j,
                                                  right_d_point[0] + right_d_point[1] * 1j)

    return (x0_c1, y0_c1, R_c1,1),(x0_c2, y0_c2, R_c2,1),x_intersection

def ellips_detection(line_detection_score_arr):
    x,y = np.meshgrid(range(line_detection_score_arr.shape[0]),range(line_detection_score_arr.shape[1]),indexing="ij")
    def elipse_score(elipse_details,args):
        """
        :param elipse_details:
        x0: center x
        y0: center y
        R: radius
        a: width is R/a
        :param args:
        sigma: length_parameter_of fit tolerance
        line_detection_score_arr: the line_detection_score_arr
        :return:
        """
        if len(elipse_details) < 4:
            temp = args[:4-len(elipse_details)]
            args = args[4 - len(elipse_details):]
            elipse_details =list(elipse_details) + temp
        x0, y0, R, a = elipse_details
        sigma = args[0]
        line_detection_score_arr = args[1]
        d = np.sqrt(a**2*(x-x0)**2 + (y-y0)**2)
        fit_value = np.dot(line_detection_score_arr.flatten(),np.exp(-np.abs(d-R)/sigma).flatten())
        return -fit_value

    # find initial guess
    (x0_e1, y0_e1, R_e1,a_e1),(x0_e2, y0_e2, R_e2,a_e2),x_intersection = make_initial_guesses_for_circles(line_detection_score_arr,x,y)

    line_detection_score_arr_right = np.array(line_detection_score_arr)
    line_detection_score_arr_right[:int(x_intersection),:] = 0

    line_detection_score_arr_left = np.array(line_detection_score_arr)
    line_detection_score_arr_left[int(x_intersection):,] = 0

    # do course optimise (sigma = 15)
    res_course1 = scipy.optimize.minimize(elipse_score, x0=np.array([x0_e1, y0_e1, R_e1]),args=[1,15,line_detection_score_arr_left], bounds=[(None,None), (None,None), (0, None)])

    # do fine optimise (sigma = 1.5)
    res_fine1 = scipy.optimize.minimize(elipse_score, np.array([*res_course1.x,1]), [1.5,line_detection_score_arr_left], bounds=[(None,None), (None,None), (0, None), (0, None)])

    # do course optimise (sigma = 15)
    res_course2 = scipy.optimize.minimize(elipse_score, x0=np.array([x0_e2, y0_e2, R_e2]),args=[1,15,line_detection_score_arr_right], bounds=[(None,None), (None,None), (0, None)])

    # do fine optimise (sigma = 1.5)
    res_fine2 = scipy.optimize.minimize(elipse_score, np.array([*res_course2.x,1]), [1.5,line_detection_score_arr_right], bounds=[(None,None), (None,None), (0, None), (0, None)])

    return res_fine1.x,res_fine2.x

def get_properties_from_file_name(name):
    name = name.replace("__","_")
    name_split = name.split("_")

    index = int(name_split[0])

    assert "mw" == name_split[1][:2]
    m_water = float(name_split[1][2:])

    assert "ms" == name_split[2][:2]
    m_soap = float(name_split[2][2:])

    assert "mp" == name_split[3][:2]
    m_weight = float(name_split[3][2:])

    if len(name_split) == 5:
        if name_split[4] == "M":
            is_garbage = True
        else:
            raise ValueError
    else:
        is_garbage = False

    part_soap = m_soap/m_water
    tension = m_weight / 2 * (9.8 * 100)
    return index, part_soap, tension, is_garbage

os.chdir(r"..\cropped photos")

# run flags - todo move somewhere else
out_file_name = r"..\analyse_data\resaults_analisys_v1_aa"
b_clean_run = False

if b_clean_run or not os.path.exists(out_file_name):
    with open(out_file_name, "w+") as out_file:
        out_file.write("photo_index, part_soap [mass ratios], tension [in dyn], is_garbage [bool], R_left [cm], a_left, R_right [cm], a_right")

with open(out_file_name, "r+") as out_file:
    lines = out_file.read().replace(" ","").split("\n")
    indexes_to_skip = [int(line.split(",")[0]) for line in lines[1:]]

b_DRAW = True
for fn in pbar(sorted(os.listdir()),position=1):
    if os.path.splitext(fn)[1] != ".png":
        continue
    if int(os.path.split(fn)[1].split("_")[0]) in indexes_to_skip:
        continue

    print(f"\nworking on {fn}")

    # extract properties from file name
    index, part_soap, tension, is_garbage = get_properties_from_file_name(os.path.splitext(os.path.split(fn)[1])[0])

    image = plt.imread(fn)
    point_f_name = os.path.join(os.path.dirname(fn), os.path.splitext(os.path.split(fn)[1])[0] + "edge_point_data.txt")
    with open(point_f_name) as point_f:
        lines = point_f.readlines()
        p0,p1,_,_ = [[float(v) for v in line.split(",")] for line in lines]

    num_of_pixels_between_markers = ((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2) ** 0.5

    print(f"\ndetecting string")
    line_detection_score_arr = score_pixel_v3p2(image).T

    print(f"\nextracting ellipses")
    try:
        (e1_x0,e1_y0,e1_R,e1_a),(e2_x0,e2_y0,e2_R,e2_a) = ellips_detection(line_detection_score_arr)
    except Exception as e:
        print("\nexception occurred")
        print(e)
        continue



    r1 = 11.9 * e1_R / num_of_pixels_between_markers
    r2 = 11.9 * e2_R / num_of_pixels_between_markers

    if b_DRAW:
        plt.figure(figsize=(12, 7))

        plt.imshow(image)
        plt.plot([p0[0], p1[0]], [p0[1], p1[1]], "r--o")

        from matplotlib.patches import Ellipse
        plt.gca().add_artist(Ellipse((e1_x0, e1_y0), e1_R*2/e1_a, e1_R*2,alpha=0.2,fill=False,edgecolor="r"))
        plt.gca().add_artist(Ellipse((e2_x0, e2_y0), e2_R*2/e2_a, e2_R*2,alpha=0.2,fill=False,edgecolor="r"))
        plt.savefig(os.path.join(r"..\analyse_data\image_results",fn),dpi=200)

        plt.close()

    print("\ndone! writing results to file \n\n")

    # write to file
    with open(out_file_name,"a+") as out_file:
        out_file.write(f"\n{index}, {part_soap:.5g}, {tension:.5g}, {is_garbage}, {r1:.5g}, {e1_a:.5g}, {r2:.5g}, {e2_a:.5g}")
plt.show()