import matplotlib.pyplot as plt
import os
import numpy as np
import scipy.ndimage
import tqdm

# I cropped the photos without writing down the location of the cropped photo in the origenal photo. so this script uses cross corelation in order to
# locate all the cropped photos in the origenal photos and writes down the locations. - runtime : around 40 minuets

b_overwrite_old = True

def to_file_format(points_arr):
    return "\n".join([", ".join(map(str, b)) for b in points_arr])

for fn in tqdm.tqdm(sorted(os.listdir("../cropped photos"))):
    fn_no_extention = os.path.splitext(os.path.split(fn)[1])[0]
    if os.path.splitext(fn)[1] != ".png":
        continue

    image_full = plt.imread(os.path.join("../labled photos",fn_no_extention + ".jpg"))

    point_f_name = os.path.join("../cropped photos", fn_no_extention + "edge_point_data.txt")
    with open(point_f_name) as point_f:
        lines = point_f.readlines()
    points_im_old = np.array([[float(v) for v in line.split(",")] for line in lines])

    crop_origin_f_name = os.path.join("../cropped photos", fn_no_extention + "crop_origin_data.txt")
    with open(crop_origin_f_name) as crop_origin_f:
        lines = crop_origin_f.readlines()
    origin_of_crop_old = np.array([float(v) for v in lines[0].split(",")])

    image_cropped_not_rotated = plt.imread(os.path.join("../cropped photos", fn_no_extention + ".png"))
    image_cropped_not_rotated_size = np.array(image_cropped_not_rotated.shape)[[1,0]]################

    crop_origin_f_name_new = fn_no_extention + "crop_origin_data.txt"

    if os.path.exists(crop_origin_f_name_new) and not b_overwrite_old:
        continue

    rotation_angle = np.arctan2(points_im_old[1][1] - points_im_old[0][1],points_im_old[1][0] - points_im_old[0][0])

    rotation_matrix = np.array([[np.cos(rotation_angle),np.sin(rotation_angle)],[-np.sin(rotation_angle),np.cos(rotation_angle)]])

    image_full_new = scipy.ndimage.rotate(image_full,rotation_angle*180/np.pi,reshape=False)

    center_of_rotation = np.array(image_full.shape)[[1,0]]/2

    apply_rotation_to_points = lambda points: center_of_rotation + np.matmul(rotation_matrix,(points - center_of_rotation).T).T

    origin_of_crop_new = apply_rotation_to_points(origin_of_crop_old)#center_of_rotation + np.matmul(rotation_matrix,origin_of_crop_old - center_of_rotation)

    original_cropped_rectangle_corners = np.array([image_cropped_not_rotated_size,
                                                   [image_cropped_not_rotated_size[0],-image_cropped_not_rotated_size[1]],
                                                   [-image_cropped_not_rotated_size[0],image_cropped_not_rotated_size[1]],
                                                   -image_cropped_not_rotated_size])/2 + origin_of_crop_old

    new_cropped_rectangle_corners = apply_rotation_to_points(original_cropped_rectangle_corners)

    rotated_rectangle_new_width_height = (np.max(new_cropped_rectangle_corners[:,0]) - np.min(new_cropped_rectangle_corners[:,0]),
                                          np.max(new_cropped_rectangle_corners[:,1]) - np.min(new_cropped_rectangle_corners[:,1]))

    image_cropped_rotated = image_full_new[
                            int(origin_of_crop_new[1] - rotated_rectangle_new_width_height[1] / 2):int(origin_of_crop_new[1] + rotated_rectangle_new_width_height[1] / 2),
                            int(origin_of_crop_new[0] - rotated_rectangle_new_width_height[0] / 2): int(origin_of_crop_new[0] + rotated_rectangle_new_width_height[0] / 2)]

    points_im_new = apply_rotation_to_points(points_im_old + origin_of_crop_old - np.array(image_cropped_not_rotated_size)/2) - origin_of_crop_new + np.array(rotated_rectangle_new_width_height)/2

    b_plot = False
    if b_plot:
        plt.figure()
        plt.imshow(image_full, alpha=0.4)
        plt.imshow(image_full_new, alpha=0.4)
        plt.scatter([origin_of_crop_old[0]], [origin_of_crop_old[1]], c="r")
        plt.scatter([origin_of_crop_new[0]], [origin_of_crop_new[1]], c="g")
        plt.scatter([center_of_rotation[0]], [center_of_rotation[1]], c="b")
        plt.scatter(original_cropped_rectangle_corners[:, 0], original_cropped_rectangle_corners[:, 1], c="gray")
        plt.scatter(new_cropped_rectangle_corners[:, 0], new_cropped_rectangle_corners[:, 1], c="m")

        plt.figure()
        plt.imshow(image_cropped_not_rotated)
        plt.scatter(points_im_old[:, 0], points_im_old[:, 1], c="r",alpha=0.6,marker="x")

        plt.figure()
        plt.imshow(image_cropped_rotated)
        plt.scatter(points_im_new[:, 0], points_im_new[:, 1], c="r")


    plt.imsave(fn,image_cropped_rotated)

    with open(fn_no_extention + "crop_origin_data.txt","w+") as out_file:
        str_to_print_to_file = to_file_format([origin_of_crop_new])
        print("writing to file:\n" + str_to_print_to_file)
        out_file.write(str_to_print_to_file)

    with open(fn_no_extention + "edge_point_data.txt","w+") as out_file:
        str_to_print_to_file = to_file_format(points_im_new)
        print("writing to file:\n" + str_to_print_to_file)
        out_file.write(str_to_print_to_file)