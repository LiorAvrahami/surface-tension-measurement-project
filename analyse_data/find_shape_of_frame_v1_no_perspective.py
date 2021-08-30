import scipy.optimize
import matplotlib.pyplot as plt
import numpy as np
import os
dist_mat = {(0,1): 119,(0,2):162,(0,3):105,(1,2):110,(1,3):158,(2,3):119}
dist_mat = {(i,j):dist_mat[i,j] if (i,j) in dist_mat else (dist_mat[j,i] if (j,i) in dist_mat else 0) for i in range(4) for j in range(4)}

def to_points(points):
    points = list(points)
    points_x = [0] + points[:3]
    points_y = [0,0] + points[3:]
    return np.array(points_x),np.array(points_y)

def score1(points,b_print = False):
    points = list(points)
    points_x,points_y = to_points(points)
    s = 0
    for i in range(4):
        for j in range(4):
            dist = ((points_x[i] - points_x[j]) ** 2 + (points_y[i] - points_y[j]) ** 2)**0.5
            s_cur = abs(dist - dist_mat[i, j])
            s += s_cur

            if b_print:
                print(f"({i},{j}):{s_cur}")
                out[(i,j)] = dist

    return s

# start optimize - find frame shape from length measuments
res = scipy.optimize.minimize(score1,[119,119,0,110,105])
p_array_mes = res.x
points_x,points_y = to_points(p_array_mes)
points_array_mes = np.array(list(zip(points_x,points_y)))

#load image data
fn = None
os.chdir("../cropped photos")
for fn in sorted(os.listdir()):
    if os.path.splitext(fn)[1] == ".png":
        break
data = plt.imread(fn)
point_f_name = os.path.join(os.path.dirname(fn), os.path.splitext(os.path.split(fn)[1])[0] + "edge_point_data.txt")
with open(point_f_name) as point_f:
    lines = point_f.readlines()

# load points from image
points_im = np.array([[float(v) for v in line.split(",")] for line in lines])
points_im = points_im[[0,1,3,2]]

# normalize image points
points_im[:,0] -= points_im[0,0]
points_im[:,1] -= points_im[0,1]


# find rotation_scaling

def score2(vec):
    global new_points
    scale_x,scale_y,rot_angle = vec
    scale_matrix = [[1/scale_x,0],[0,1/scale_y]]
    rot_matrix = [[np.cos(rot_angle),-np.sin(rot_angle)],[np.sin(rot_angle),np.cos(rot_angle)]]
    new_points = np.linalg.multi_dot([scale_matrix,rot_matrix,points_im.T]).T
    score = np.sum((new_points - points_array_mes)**2)
    return score

res2 = scipy.optimize.minimize(score2,[1,1,0],bounds=[(0,np.inf),(0,np.inf),(-1,1)])
scale_x,scale_y,rot_angle = res2.x
score2([scale_x,scale_y,rot_angle])
points_im_scaled = new_points

p_array = [points_im_scaled[1,0],points_im_scaled[2,0],points_im_scaled[3,0],points_im_scaled[2,1],points_im_scaled[3,1]]
points_x_im, points_y_im = to_points(p_array)

out = {}

plt.plot(list(points_x) + [points_x[0]],list(points_y) + [points_y[0]],"-o")
plt.plot(list(points_x_im) + [points_x_im[0]],list(points_y_im) + [points_y_im[0]],"-o")
plt.grid(ls="--")

score1(p_array_mes, True)





