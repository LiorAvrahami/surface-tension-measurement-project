import os
import numpy as np
import scipy.optimize

frame_points_real = np.array([[-5.947945281, -5.374567474], [5.944712097, -5.374567474], [-5.942513822, 5.12370247], [5.945747006, 5.625432479]])


def get_image_and_real_coordinates_converters(frame_points_in_image, cropped_photo_origin, cropped_photo_size):
    """
    :param frame_points_in_image:
            the four points on the frame that were pointed out for each photo file, these are also stored in an adgacent file:
            photo_file_name_no_extension + "edge_point_data.txt"
    :param cropped_photo_origin: the location of the middle of the current cropped photo, inside the original full sized photo.
        this data was already calculated for each photo file, and was placed in photo_file_name_no_extension + "crop_origin_data.txt"
    :param cropped_photo_size: the width and height of the cropped_photo.
    :return:
        a pair of function objects that transforms points from the real coordinate system to the image coordinate system, and visa versa.
        functions are returned in a tuple, the first is the real to image converter, and the second is the image to real converter.
        return functions signatures are:
            functor_real_to_image(numpy.ndarray real_system_points_to_transform) -> the_transformed_points_in_iamge_coordinates
            functor_image_to_real(numpy.ndarray image_system_points_to_transform) -> the_transformed_points_in_real_coordinates
    """
    frame_points_in_image = cropped_photo_coordinates_to_full_photo_coordinates(frame_points_in_image, cropped_photo_origin, cropped_photo_size)
    transform_arguments = find_perspective_transform_arguments(frame_points_in_image)

    def image_to_real_coordinate_transform(image_system_points_to_transform):
        image_system_points_to_transform = cropped_photo_coordinates_to_full_photo_coordinates(image_system_points_to_transform, cropped_photo_origin, cropped_photo_size)
        return convert_image_points_to_real_points(transform_arguments, image_system_points_to_transform)

    def real_to_image_coordinate_transform(real_system_points_to_transform):
        image_system_points = convert_real_points_to_image_points(transform_arguments, real_system_points_to_transform)
        image_system_points = full_photo_coordinates_to_cropped_photo_coordinates(image_system_points, cropped_photo_origin, cropped_photo_size)
        return image_system_points

    return (real_to_image_coordinate_transform,image_to_real_coordinate_transform,transform_arguments)



def cropped_photo_coordinates_to_full_photo_coordinates(points_in_cropped_coor, cropped_photo_origin, cropped_photo_size):
    return points_in_cropped_coor + cropped_photo_origin - np.array([3036 / 2, 4048 / 2]) - np.array(
        [cropped_photo_size[1] / 2, cropped_photo_size[0] / 2])

def full_photo_coordinates_to_cropped_photo_coordinates(points_in_full_image_coor, cropped_photo_origin, cropped_photo_size):
    return points_in_full_image_coor - cropped_photo_origin + np.array([3036 / 2, 4048 / 2]) + np.array(
        [cropped_photo_size[1] / 2, cropped_photo_size[0] / 2])


def find_perspective_transform_arguments(frame_points_in_image):
    """
    use scipy.optimise to find the 3d geometry of the frame (and scaling)
    :param frame_points_in_image:
    :return: transform_arguments
    """
    frame_points_real_affine = np.append(frame_points_real, np.ones((frame_points_real.shape[0], 1)), axis=1)
    frame_points_in_image_affine = np.append(frame_points_in_image, np.ones((frame_points_in_image.shape[0], 1)), axis=1)

    res = global_minimize_scale(fun=score_perspective_transform_arguments,
                          bounds=[(1, np.inf), (-0.7, 0.7), (-0.7, 0.7), (-np.pi / 4, np.pi / 4), (-np.inf, np.inf),(-np.inf, np.inf)],
                          args=[frame_points_real_affine, frame_points_in_image_affine])
    transform_arguments = res.x
    return transform_arguments


def global_minimize_scale(fun,bounds,args,min_power=1,max_power=5,num_of_try=50,maximal_score_to_succeed=100):
    best_res = scipy.optimize.minimize(fun=fun,x0=np.array([3400, 0, 0, 0, 0, 0]),bounds=bounds,args=args)
    for a0 in np.logspace(min_power,max_power,num_of_try):
        res = scipy.optimize.minimize(fun=fun,x0=np.array([a0, 0, 0, 0, 0, 0]),bounds=bounds,args=args)
        if res.fun < best_res.fun:
            best_res = res
    if best_res.fun > maximal_score_to_succeed:
        raise RuntimeError("could not find perspective parameters")
    return best_res

def score_perspective_transform_arguments(perspective_transform_arguments, args):
    frame_points_real_affine, frame_points_in_image_affine = args
    new_points = do_transform_fast(perspective_transform_arguments, frame_points_real_affine)
    score = np.sum((new_points - frame_points_in_image_affine[:,:2]) ** 2)
    return np.sqrt(score)


# find rotation_scaling
def do_transform_fast(transform_arguments, frame_points_real_affine):
    scale, z1, z2, rot_angle, Tx, Ty = transform_arguments
    z0 = 40  # the photos were take n at about 400 mm away

    rot_matrix = [[np.cos(rot_angle), -np.sin(rot_angle), 0], [np.sin(rot_angle), np.cos(rot_angle), 0], [0, 0, 1]]
    Plane_matrix = [[np.sqrt(1 - z1 ** 2), 0, 0], [0, np.sqrt(1 - z2 ** 2), 0], [z1, z2, z0]]
    Translate_matrix = [[1, 0, Tx], [0, 1, Ty], [0, 0, 1]]

    new_points = np.linalg.multi_dot([Plane_matrix, Translate_matrix, rot_matrix, frame_points_real_affine.T]).T

    new_points[:, 0] /= new_points[:, 2]
    new_points[:, 1] /= new_points[:, 2]
    new_points = new_points[:, :2] * scale
    return new_points

def convert_real_points_to_image_points(transform_arguments, frame_points_real):
    if frame_points_real is not np.ndarray:
        frame_points_real = np.array(frame_points_real)
    frame_points_real_affine = np.append(frame_points_real, np.ones((frame_points_real.shape[0], 1)), axis=1)
    return do_transform_fast(transform_arguments,frame_points_real_affine)

def convert_image_points_to_real_points(transform_arguments, image_system_points):
    scale, z1, z2, rot_angle, Tx, Ty = transform_arguments
    z0 = 40  # the photos were take n at about 400 mm away
    if image_system_points is not np.ndarray:
        image_system_points = np.array(image_system_points)

    S1, S2 = np.sqrt(1 - z1 ** 2), np.sqrt(1 - z2 ** 2)
    rot_matrix = [[np.cos(-rot_angle), -np.sin(-rot_angle), 0], [np.sin(-rot_angle), np.cos(-rot_angle), 0], [0, 0, 1]]
    Plane_matrix = [[z0 / S1, 0, 0], [0, z0 / S2, 0], [-z1 / S1, -z2 / S2, 1]]
    Translate_matrix = [[1, 0, -Tx], [0, 1, -Ty], [0, 0, 1]]

    image_system_points_affin = np.append(image_system_points, np.ones((image_system_points.shape[0], 1)), axis=1)
    image_system_points_affin[:, :2] /= scale

    new_points = np.linalg.multi_dot([Plane_matrix, image_system_points_affin.T]).T

    new_points[:, 0] /= new_points[:, 2]
    new_points[:, 1] /= new_points[:, 2]
    new_points[:, 2] = 1

    new_points = np.linalg.multi_dot([rot_matrix, Translate_matrix, new_points.T]).T

    return new_points[:,:2]
