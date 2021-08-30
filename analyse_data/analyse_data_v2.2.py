import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import os
from line_detection import score_pixel_v3p2
from scipy.ndimage import map_coordinates
import scipy.optimize
import numpy as np
from tqdm import tqdm as pbar
from tqdm.contrib import itertools as pbar_itertools
from undo_perspective_transform_on_image_points import get_image_and_real_coordinates_converters

def real_coordinates_circle_detection(line_detection_score_arr,image_to_real_coordinate_transform,real_to_image_coordinate_transform,b_draw = False,save_figures_path=None):
    # find initial guess
    (x0_e1, y0_e1, R_e1),(x0_e2, y0_e2, R_e2),x_intersection = make_initial_guesses_for_circles(image_to_real_coordinate_transform,line_detection_score_arr)

    def fit_circle(points_x,points_y,weights,x0e,y0e,Re,is_left_side):
        weights = np.maximum(0, weights)
        weights = weights / max(weights)
        indexes_to_delete = weights <= 0
        weights = weights[~indexes_to_delete]
        points_x = points_x[~indexes_to_delete]
        points_y = points_y[~indexes_to_delete]
        if b_draw:
            plt.figure(figsize=(12, 7))
            plt.imshow(line_detection_score_arr.T)
            points_image = real_to_image_coordinate_transform(np.append(points_x[:,np.newaxis],points_y[:,np.newaxis],axis=1))
            plt.scatter(points_image[:,0], points_image[:,1], c="r", alpha=weights*0.1)
        # to polar
        theta = np.arctan2(points_x-x0e,points_y-y0e)
        r = np.sqrt((points_x-x0e)**2 + (points_y-y0e)**2)

        def circle_in_polar(theta, x0pol, y0pol, R):
            return x0pol * np.sin(theta) + y0pol * np.cos(theta) + np.sqrt(
                (x0pol * np.sin(theta) + y0pol * np.cos(theta)) ** 2 + (R ** 2 - x0pol ** 2 - y0pol ** 2))

        # remove_outliers - mess around with weights
        mean_r = np.sum(r * weights) / np.sum(weights)
        weights = weights / (abs(r-mean_r)**2 + 1)
        weights /= max(weights)

        (x0pol, y0pol, R), (cov) = scipy.optimize.curve_fit(circle_in_polar, theta, r, [0, 0, Re], sigma=1 / weights,maxfev=3000)
        R = abs(R)

        if b_draw:
            # print(f"R = {11.9 * R / num_of_pixels_between_markers:4g}cm")
            # plt.gca().add_artist(Ellipse((x0e, y0e), Re * 2, Re * 2, alpha=1, fill=False, color="b"))
            # plt.gca().add_artist(Ellipse((x0e + x0pol, y0e + y0pol), R * 2, R * 2, alpha=1, fill=False, color="r"))
            draw_circle_fit_on_image(real_to_image_coordinate_transform,Re,x0e,y0e,color="b",alpha=0.5,linewidth=0.5)
            draw_circle_fit_on_image(real_to_image_coordinate_transform, R, x0e + x0pol, y0e + y0pol,color="r",alpha=0.5,linewidth=0.5)

            add_to_name = ("left" if is_left_side else "right") + "fit_on_image" + ".png"
            plt.savefig(os.path.join(r"..\analyse_data\image_results_v2", save_figures_path + add_to_name), dpi=200)
            plt.close()

            plt.figure(figsize=(12, 7))
            plt.scatter(theta, r, alpha=weights)
            plt.scatter(theta + 2 * np.pi, r, alpha=weights)
            tetha_th = np.linspace(0, 12, 1000)
            plt.plot(tetha_th, circle_in_polar(tetha_th, x0pol, y0pol, R), "r",alpha=0.5,linewidth=0.5)
            plt.plot(tetha_th, circle_in_polar(tetha_th, 0, 0, Re), "g",alpha=0.5,linewidth=0.5)

            add_to_name = ("left" if is_left_side else "right") + "fit_in_polar" + ".png"
            plt.savefig(os.path.join(r"..\analyse_data\image_results_v2", save_figures_path + add_to_name), dpi=200)
            plt.close()

        return (x0pol + x0e, y0pol + y0e, R)

    horizontal_scan_points_y = np.argmax(line_detection_score_arr, 1)
    horizontal_scan_points_x = np.arange(len(horizontal_scan_points_y))

    vertical_scan_points_x_left, vertical_scan_points_x_right = zip(*[get_two_largest_local_maxima(line_detection_score_arr[:, i]) for i in range(line_detection_score_arr.shape[1])])
    vertical_scan_points_y_left, vertical_scan_points_y_right = np.arange(len(vertical_scan_points_x_left)), np.arange(len(vertical_scan_points_x_right))

    vertical_scan_points_x_left = np.array(vertical_scan_points_x_left)
    vertical_scan_points_x_right = np.array(vertical_scan_points_x_right)


    vertical_scan_points_y_left = vertical_scan_points_y_left[vertical_scan_points_x_left < x_intersection]
    vertical_scan_points_x_left = vertical_scan_points_x_left[vertical_scan_points_x_left < x_intersection]
    vertical_scan_points_y_right = vertical_scan_points_y_right[vertical_scan_points_x_right > x_intersection]
    vertical_scan_points_x_right = vertical_scan_points_x_right[vertical_scan_points_x_right > x_intersection]

    xleft = np.concatenate([vertical_scan_points_x_left, horizontal_scan_points_x[:int(x_intersection)]])
    yleft = np.concatenate([vertical_scan_points_y_left,horizontal_scan_points_y[:int(x_intersection)]])
    points_left = np.append(xleft[:,np.newaxis],yleft[:,np.newaxis],axis=1)
    weights_left = line_detection_score_arr[xleft, yleft]
    xright = np.concatenate([vertical_scan_points_x_right, horizontal_scan_points_x[int(x_intersection):]])
    yright = np.concatenate([vertical_scan_points_y_right, horizontal_scan_points_y[int(x_intersection):]])
    points_right = np.append(xright[:,np.newaxis],yright[:,np.newaxis],axis=1)
    weights_right = line_detection_score_arr[xright, yright]

    points_left = image_to_real_coordinate_transform(points_left)
    points_right = image_to_real_coordinate_transform(points_right)

    try:
        res1 = fit_circle(points_left[:,0],points_left[:,1],weights_left,x0_e1,y0_e1,R_e1,is_left_side=True)
    except:
        res1 = x0_e1,y0_e1,R_e1
    try:
        res2 = fit_circle(points_right[:,0],points_right[:,1],weights_right,x0_e2,y0_e2,R_e2,is_left_side=False)
    except:
        res2 = x0_e2,y0_e2,R_e2
    return res1,res2

def make_initial_guesses_for_circles_old(image_to_real_coordinate_transform,line_detection_score_arr):
    x, y = np.meshgrid(range(line_detection_score_arr.shape[0]), range(line_detection_score_arr.shape[1]), indexing="ij")
    sx,sy = line_detection_score_arr.shape
    line_detection_score_arr = line_detection_score_arr**3
    low_image = line_detection_score_arr[:,int(sy*0.7):]
    low_x = x[:, int(sy * 0.7):]

    horizontal_scan_points_y = np.argmax(line_detection_score_arr, 1)
    plt.plot(np.convolve(horizontal_scan_points_y, np.linspace(-1, 1, 10)))

    x_intersection = split_image_horizontally(line_detection_score_arr)
    x_intersection_point = image_to_real_coordinate_transform([[x_intersection,sy]])[0]

    left_image = line_detection_score_arr[:int(sx*0.2)]
    left_y = y[:int(sx*0.2)]
    y_left_intersection = np.sum(left_y * left_image) / np.sum(left_image)
    y_left_intersection_point = image_to_real_coordinate_transform([[0, y_left_intersection]])[0]

    right_image = line_detection_score_arr[int(sx*0.8):]
    right_y = y[int(sx*0.8):]
    y_right_intersection = np.sum(right_y * right_image) / np.sum(right_image)
    y_right_intersection_point = image_to_real_coordinate_transform([[sx, y_right_intersection]])[0]

    diag_left_mask = line_detection_score_arr*np.exp(-np.abs(x + (y-sy)))
    left_d_point = np.sum(x*diag_left_mask)/np.sum(diag_left_mask), np.sum(y*diag_left_mask)/np.sum(diag_left_mask)
    left_d_point = image_to_real_coordinate_transform([left_d_point])[0]

    diag_right_mask = line_detection_score_arr*np.exp(-np.abs((x-sx) - (y-sy)))
    right_d_point = np.sum(x * diag_right_mask) / np.sum(diag_right_mask), np.sum(y * diag_right_mask) / np.sum(diag_right_mask)
    right_d_point = image_to_real_coordinate_transform([right_d_point])[0]

    x0_c1, y0_c1, R_c1 = get_circle_from_3_points(x_intersection_point[0] + x_intersection_point[1] * 1j,
                                                  y_left_intersection_point[0] + y_left_intersection_point[1] * 1j,
                                                  left_d_point[0] + left_d_point[1] * 1j)

    x0_c2, y0_c2, R_c2 = get_circle_from_3_points(x_intersection_point[0] + x_intersection_point[1] * 1j,
                                                  y_right_intersection_point[0] + y_right_intersection_point[1] * 1j,
                                                  right_d_point[0] + right_d_point[1] * 1j)

    return (x0_c1, y0_c1, R_c1),(x0_c2, y0_c2, R_c2),x_intersection

def make_initial_guesses_for_circles(image_to_real_coordinate_transform,line_detection_score_arr):
    x, y = np.meshgrid(range(line_detection_score_arr.shape[0]), range(line_detection_score_arr.shape[1]), indexing="ij")
    sx,sy = line_detection_score_arr.shape
    line_detection_score_arr = line_detection_score_arr**3

    x_intersection = split_image_horizontally(line_detection_score_arr)
    x_intersection_point = image_to_real_coordinate_transform([[x_intersection,sy]])[0]

    left_image = line_detection_score_arr[:int(sx*0.2)]
    left_y = y[:int(sx*0.2)]
    y_left_intersection = np.sum(left_y * left_image) / np.sum(left_image)
    y_left_intersection_point = image_to_real_coordinate_transform([[0, y_left_intersection]])[0]

    right_image = line_detection_score_arr[int(sx*0.8):]
    right_y = y[int(sx*0.8):]
    y_right_intersection = np.sum(right_y * right_image) / np.sum(right_image)
    y_right_intersection_point = image_to_real_coordinate_transform([[sx, y_right_intersection]])[0]

    diag_left_mask = line_detection_score_arr*np.exp(-np.abs(x + (y-sy)))
    left_d_point = np.sum(x*diag_left_mask)/np.sum(diag_left_mask), np.sum(y*diag_left_mask)/np.sum(diag_left_mask)
    left_d_point = image_to_real_coordinate_transform([left_d_point])[0]

    diag_right_mask = line_detection_score_arr*np.exp(-np.abs((x-sx) - (y-sy)))
    right_d_point = np.sum(x * diag_right_mask) / np.sum(diag_right_mask), np.sum(y * diag_right_mask) / np.sum(diag_right_mask)
    right_d_point = image_to_real_coordinate_transform([right_d_point])[0]

    x0_c1, y0_c1, R_c1 = get_circle_from_3_points(x_intersection_point[0] + x_intersection_point[1] * 1j,
                                                  y_left_intersection_point[0] + y_left_intersection_point[1] * 1j,
                                                  left_d_point[0] + left_d_point[1] * 1j)

    x0_c2, y0_c2, R_c2 = get_circle_from_3_points(x_intersection_point[0] + x_intersection_point[1] * 1j,
                                                  y_right_intersection_point[0] + y_right_intersection_point[1] * 1j,
                                                  right_d_point[0] + right_d_point[1] * 1j)

    min_x, max_y1 = image_to_real_coordinate_transform([0,line_detection_score_arr.shape[1]])[0]
    max_x, max_y2 = image_to_real_coordinate_transform([line_detection_score_arr.shape[0], line_detection_score_arr.shape[1]])[0]
    max_y = (max_y1 + max_y2)/2
    max_r1 = np.sqrt((x_intersection_point[0] - min_x) ** 2 + (x_intersection_point[1] - max_y) ** 2)
    max_r2 = np.sqrt((x_intersection_point[0] - max_x) ** 2 + (x_intersection_point[1] - max_y) ** 2)

    # x0_c1 = min(max_x,max(min_x,x0_c1))
    # y0_c2 = min(max_y,y0_c2)
    # x0_c2 = min(max_x,max(min_x,x0_c2))
    # y0_c1 = min(max_y, y0_c1)
    # R_c1 = min(max_r1, R_c1)
    # R_c2 = min(max_r2, R_c2)

    x0_c1 = min_x
    y0_c2 = max_y
    x0_c2 = max_x
    y0_c1 = max_y
    R_c1 = max_r1
    R_c2 = max_r2

    return (x0_c1, y0_c1, R_c1),(x0_c2, y0_c2, R_c2),x_intersection

def split_image_horizontally(line_detection_score_arr,window_size = 700):
    argmax_gradient = np.convolve(np.argmax(line_detection_score_arr, 1), np.linspace(-1, 1, 10))
    argmax_gradient = np.minimum(np.maximum(argmax_gradient,-100),100)

    def score_x_intersection(x_intersection_candidate):
        x_intersection_candidate = int(x_intersection_candidate)
        return np.sum(argmax_gradient[x_intersection_candidate - window_size//2:x_intersection_candidate] - argmax_gradient[x_intersection_candidate:x_intersection_candidate + window_size//2])

    res = scipy.optimize.minimize_scalar(score_x_intersection,bounds=(window_size//2 + 1,line_detection_score_arr.shape[0]-window_size//2 + 1),method="Bounded")
    return res.x

def remove_outliers_in_polar_space(x,y):
    pass

def get_two_largest_local_maxima(profile):
    temp = abs(profile[1:-1] * np.diff(np.sign(np.diff(profile))))
    idx1 = np.argmax(temp)
    temp[idx1] = 0
    idx2 = np.argmax(temp)
    return min(idx1, idx2), max(idx1, idx2)

def get_circle_from_3_points(p1,p2,p3):
    """
    :param p1,p2,p3: complex numbers representing a point on the xy plane. (y is imaginary part)
    :return: x0, y0, R
    """
    w = p3 - p1
    w /= p2 - p1
    c = (p1 - p2) * (w - abs(w) ** 2) / 2j / w.imag - p1
    return -np.real(c),-np.imag(c),np.absolute(c + p1)

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
    mg = m_weight * (9.8 * 100)
    return index, part_soap, mg, is_garbage

def draw_circle_fit_on_image(real_to_image_coordinate_transform,radius,x0,y0,axes_to_draw_on=None,**kw_args_plot):
    if axes_to_draw_on is None:
        axes_to_draw_on = plt.gca()
    theta = np.linspace(0,2*np.pi,100,endpoint=True)
    points_real = np.append((x0 + radius*np.sin(theta))[:,np.newaxis],((y0 + radius*np.cos(theta))[:,np.newaxis]),axis=1)
    points_image = real_to_image_coordinate_transform(points_real)
    axes_to_draw_on.plot(points_image[:,0],points_image[:,1],"-",**kw_args_plot)

os.chdir(r"..\cropped photos")

# run flags - todo move somewhere else
out_file_name = r"..\analyse_data\resaults_analisys_v2.2"
b_clean_run = True
b_DRAW = True

if b_clean_run or not os.path.exists(out_file_name):
    with open(out_file_name, "w+") as out_file:
        out_file.write("photo_index, mg [in dyn], distance_between_centers_of_circles [cm], part_soap [mass ratios], is_garbage [bool], R_left [cm], x_center_left [cm], y_center_left [cm], R_right [cm], x_center_right [cm], y_center_right [cm]")

with open(out_file_name, "r+") as out_file:
    lines = out_file.read().replace(" ","").split("\n")
    indexes_to_skip = [int(line.split(",")[0]) for line in lines[1:]]

for fn in pbar(sorted(os.listdir()),position=1):
    fn_no_extention = os.path.splitext(os.path.split(fn)[1])[0]
    if os.path.splitext(fn)[1] != ".png":
        continue
    if int(os.path.split(fn)[1].split("_")[0]) in indexes_to_skip:
        continue

    print(f"\nworking on {fn}")

    image = plt.imread(fn)

    # extract properties from file name
    index, part_soap, mg, is_garbage = get_properties_from_file_name(fn_no_extention)

    #
    point_f_name = os.path.join(os.path.dirname(fn), fn_no_extention + "edge_point_data.txt")
    with open(point_f_name) as point_f:
        lines = point_f.readlines()
    points_im = np.array([[float(v) for v in line.split(",")] for line in lines])

    crop_origin_f_name = os.path.join(os.path.dirname(fn), fn_no_extention + "crop_origin_data.txt")
    with open(crop_origin_f_name) as crop_origin_f:
        lines = crop_origin_f.readlines()
    origin_of_image_points = np.array([[float(v) for v in line.split(",")] for line in lines])

    #
    real_to_image_coordinate_transform,image_to_real_coordinate_transform, transform_arguments = get_image_and_real_coordinates_converters(points_im,origin_of_image_points,image.shape)

    print(f"\ndetecting string")
    line_detection_results_file_name = os.path.join(r"..\analyse_data\line_detection_results",fn_no_extention + ".npy")
    if os.path.exists(line_detection_results_file_name):
        line_detection_score_arr = np.load(line_detection_results_file_name)
    else:
        line_detection_score_arr = score_pixel_v3p2(image).T
        np.save(line_detection_results_file_name,line_detection_score_arr)

    print(f"\nextracting ellipses")
    (e1_x0,e1_y0,r1),(e2_x0,e2_y0,r2) = real_coordinates_circle_detection(line_detection_score_arr,image_to_real_coordinate_transform,real_to_image_coordinate_transform,b_draw=b_DRAW,save_figures_path=fn_no_extention)

    if b_DRAW:
        plt.figure(figsize=(12, 7))

        plt.imshow(image)
        draw_circle_fit_on_image(real_to_image_coordinate_transform, r1, e1_x0, e1_y0, color="r",linewidth=0.5,alpha=0.5)
        draw_circle_fit_on_image(real_to_image_coordinate_transform, r2, e2_x0, e2_y0, color="r",linewidth=0.5,alpha=0.5)
        plt.savefig(os.path.join(r"..\analyse_data\image_results_v2",fn),dpi=200)
        plt.close()

    print("\ndone! writing results to file \n\n")

    # write to file
    with open(out_file_name,"a+") as out_file:
        out_file.write(f"\n{index}, {mg:.5g}, {np.sqrt((e1_x0-e2_x0)**2 + (e1_y0-e2_y0)**2):.5g},{part_soap:.5g}, {is_garbage}, {r1:.5g}, {e1_x0:.5g}, {e1_y0:.5g}, {r2:.5g}, {e2_x0:.5g}, {e2_y0:.5g}")
plt.show()