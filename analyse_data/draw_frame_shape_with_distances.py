import scipy.optimize
import matplotlib.pyplot as plt
import numpy as np
import os
import addcopyfighandler

dist_mat = {(0,1): 119,(0,2):162,(0,3):105,(1,2):110,(1,3):158,(2,3):119}
dist_mat = {(i,j):dist_mat[i,j] if (i,j) in dist_mat else (dist_mat[j,i] if (j,i) in dist_mat else 0) for i in range(4) for j in range(4)}
frame_points_real = np.array([[-5.947945281, -5.374567474], [5.944712097, -5.374567474], [-5.942513822, 5.12370247], [5.945747006, 5.625432479]])

fn = r"../cropped photos/01_mw552.9_ms2.48_mp0.525.png"
fn_no_extention = os.path.splitext(fn)[0]

point_f_name = os.path.join("../cropped photos", fn_no_extention + "edge_point_data.txt")
with open(point_f_name) as point_f:
    lines = point_f.readlines()
points_im_old = np.array([[float(v) for v in line.split(",")] for line in lines])

image_cropped = plt.imread(fn)

plt.figure()
plt.imshow(image_cropped)
plt.scatter(points_im_old[:, 0], points_im_old[:, 1], c="r", alpha=0.9, marker="x")

for i in range(4):
    for j in range(i + 1,4):
        p1 = points_im_old[i]
        p2 = points_im_old[j]

        text_position_factor = 0.6
        text_position = (p1 * text_position_factor + p2 * (1-text_position_factor))

        trim_factor = 0.95
        p1 = p1 * trim_factor + p2 * (1 - trim_factor)
        p2 = p2 * trim_factor + p1 * (1 - trim_factor)
        angle = (np.arctan2(*((p2-p1)[[1,0]]))*180/np.pi + 90) % 180 - 90
        plt.plot([p1[0],p2[0]],[p1[1],p2[1]],"r--")
        plt.text(text_position[0], text_position[1], str(dist_mat[i, j] / 10) + "[cm]", {"fontsize":10.5, "fontweight": 'bold', 'bbox':{'fc': '0.8', 'pad': 1}, 'ha': 'center', 'va': 'center'}, color=(0.6, 0.1, 0.1), rotation=-angle)

plt.figure()
plt.imshow(image_cropped)
plt.scatter(points_im_old[:, 0], points_im_old[:, 1], c="r", alpha=0.9, marker="x")
for i in range(4):
    plt.text(points_im_old[i,0], points_im_old[i,1] - 60, f"({frame_points_real[i][0]:.3g},{frame_points_real[i][1]:.3g}) [cm]", {"fontsize":10.5, "fontweight": 'bold', 'bbox':{'fc': '0.8', 'pad': 1}, 'ha': 'center', 'va': 'center'}, color=(0.6, 0.1, 0.1), rotation=-angle)