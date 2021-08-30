import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.signal import correlate
from scipy.ndimage import gaussian_filter
import tqdm

# I cropped the photos without writing down the location of the cropped photo in the origenal photo. so this script uses cross corelation in order to
# locate all the cropped photos in the origenal photos and writes down the locations. - runtime : around 40 minuets

b_overwrite_old = False

def to_file_format(points_arr):
    return "\n".join([", ".join(map(str, b)) for b in points_arr])


def calculate_normalised_crosscorrelation(document_image,char_image):
    document_image = document_image - np.mean(document_image)
    char_image = char_image - np.mean(char_image)
    crcr = correlate(document_image, char_image, mode="same")
    norm = correlate(document_image**2, char_image*0 + 1, mode="same")**0.5
    fitv = crcr/norm
    return fitv

for fn in tqdm.tqdm(sorted(os.listdir())):
    fn_no_extention = os.path.splitext(os.path.split(fn)[1])[0]
    if os.path.splitext(fn)[1] != ".png":
        continue
    data_cropped = plt.imread(fn)
    data_full = plt.imread(os.path.join("../labled photos",fn_no_extention + ".jpg"))
    crop_origin_f_name = os.path.join(os.path.dirname(fn), fn_no_extention + "crop_origin_data.txt")
    if os.path.exists(crop_origin_f_name) and not b_overwrite_old:
        continue

    ccr_r = calculate_normalised_crosscorrelation(data_full[:, :, 0], data_cropped[:, :, 0])
    ccr_g = calculate_normalised_crosscorrelation(data_full[:, :, 1], data_cropped[:, :, 1])
    ccr_b = calculate_normalised_crosscorrelation(data_full[:, :, 2], data_cropped[:, :, 2])
    ccr = ccr_r + ccr_g + ccr_b
    ccr = ccr - gaussian_filter(ccr, 50)

    origin_point = np.unravel_index(np.argmax(ccr),ccr.shape)

    b_plot = False
    if b_plot:
        plt.figure()
        plt.imshow(data_full, alpha=0.4)
        xlim, ylim = plt.xlim(), plt.ylim()
        plt.imshow(data_cropped, alpha=0.4,
                   extent=[origin_point[1] - data_cropped.shape[1] / 2, origin_point[1] + data_cropped.shape[1] / 2,
                           origin_point[0] + data_cropped.shape[0] / 2, origin_point[0] - data_cropped.shape[0] / 2])
        plt.xlim(xlim), plt.ylim(ylim)

    origin_point = [origin_point[1],origin_point[0]]


    with open(crop_origin_f_name,"w+") as out_file:
        str_to_print_to_file = to_file_format([origin_point])
        print("writing to file:\n" + str_to_print_to_file)
        out_file.write(str_to_print_to_file)