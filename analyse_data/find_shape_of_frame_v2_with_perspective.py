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

def to_points2(points):
    temp = [points[1, 0], points[2, 0], points[3, 0], points[2, 1], points[3, 1]]
    return to_points(temp)

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
# load points from image
point_f_name = os.path.join(os.path.dirname(fn), os.path.splitext(os.path.split(fn)[1])[0] + "edge_point_data.txt")
with open(point_f_name) as point_f:
    lines = point_f.readlines()
points_im = np.array([[float(v) for v in line.split(",")] for line in lines])
points_im = points_im[[0,1,3,2]]
# load center of cropped_image in the original_image_coordinates
point_f_name = os.path.join(os.path.dirname(fn), os.path.splitext(os.path.split(fn)[1])[0] + "crop_origin_data.txt")
with open(point_f_name) as point_f:
    lines = point_f.readlines()
origin_of_image_points = np.array([[float(v) for v in line.split(",")] for line in lines])

points_im += origin_of_image_points - np.array([3036/2,4048/2]) - np.array([data.shape[1]/2,data.shape[0]/2])


points_array_mes_affine = np.append(points_array_mes,np.ones((points_array_mes.shape[0],1)),axis=1)
points_array_mes_affine[:,0] -= np.mean(points_array_mes_affine[:,0])
points_array_mes_affine[:,1] -= np.mean(points_array_mes_affine[:,1])


# find rotation_scaling
def do_transform(vec,points_array_mes_affine):
    scale,z1,z2,rot_angle,Tx,Ty = vec
    z0 = 400 # the photos were take n at about 400 mm away
    rot_matrix = [[np.cos(rot_angle), -np.sin(rot_angle),0], [np.sin(rot_angle), np.cos(rot_angle),0],[0,0,1]]
    Plane_matrix = [[np.sqrt(1-z1**2),0,0],[0,np.sqrt(1-z2**2),0],[z1,z2,z0]]
    Translate_matrix = [[1,0,Tx],[0,1,Ty],[0,0,1]]

    new_points = np.linalg.multi_dot([Plane_matrix,Translate_matrix,rot_matrix,points_array_mes_affine.T]).T

    new_points[:, 0] /= new_points[:, 2]
    new_points[:, 1] /= new_points[:, 2]
    new_points = new_points[:,:2]*scale
    return new_points

def do_reverse_transform(vec,image_system_points):
    scale,z1,z2,rot_angle,Tx,Ty = vec
    z0 = 400 # the photos were take n at about 400 mm away
    S1,S2 = np.sqrt(1-z1**2),np.sqrt(1-z2**2)
    rot_matrix = [[np.cos(-rot_angle), -np.sin(-rot_angle),0], [np.sin(-rot_angle), np.cos(-rot_angle),0],[0,0,1]]
    Plane_matrix = [[z0/S1,0,0],[0,z0/S2,0],[-z1/S1,-z2/S2,1]]
    Translate_matrix = [[1,0,-Tx],[0,1,-Ty],[0,0,1]]

    image_system_points_affin = np.append(image_system_points,np.ones((image_system_points.shape[0],1)),axis=1)
    image_system_points_affin[:,:2] /= scale

    new_points = np.linalg.multi_dot([Plane_matrix,image_system_points_affin.T]).T

    new_points[:, 0] /= new_points[:, 2]
    new_points[:, 1] /= new_points[:, 2]
    new_points[:, 2] = 1

    new_points = np.linalg.multi_dot([rot_matrix, Translate_matrix, new_points.T]).T

    return new_points

def score2(vec):
    new_points = do_transform(vec,points_array_mes_affine)
    score = np.sum((new_points - points_im)**2)
    return np.sqrt(score)

res2 = scipy.optimize.minimize(score2,np.array([1,0,0,0,0,0]),bounds=[(1,np.inf),(-1,1),(-1,1),(-np.pi/4,np.pi/4),(-np.inf,np.inf),(-np.inf,np.inf)])
# scale,z1,z2,rot_angle = res2.x

points_x_im, points_y_im = points_im[:,0], points_im[:,1]
points_array_mes_transformed = do_transform(res2.x,points_array_mes_affine)
points_x,points_y = points_array_mes_transformed[:,0], points_array_mes_transformed[:,1]
#plt.plot(list(points_x) + [points_x[0]],list(points_y) + [points_y[0]],"-o")

out = {}
plt.figure()
plt.plot(list(points_x) + [points_x[0]],list(points_y) + [points_y[0]],"-o",label="perspective applied to measured points")
plt.plot(list(points_x_im) + [points_x_im[0]],list(points_y_im) + [points_y_im[0]],"-o",label="from image")
# plt.imshow(data)
plt.grid(ls="--")
plt.axis("equal")
plt.legend()

plt.figure()
prrr = do_reverse_transform(res2.x,points_array_mes_transformed)
points_im_revtransformed = do_reverse_transform(res2.x,points_im)
plt.plot(points_array_mes_affine[:,0],points_array_mes_affine[:,1],'-o')
plt.plot(prrr[:,0],prrr[:,1],'-o')
plt.plot(points_im_revtransformed[:,0],points_im_revtransformed[:,1],'-o')

points1 = np.random.uniform(-1000,100,(1000,2))
# points[:,1] = 25
vec = [1,0,0,0,100,100]
plt.figure()
plt.scatter(points1[:,0],points1[:,1],alpha=0.7)

points_affin = np.append(points1,np.ones((points1.shape[0],1)),axis=1)
points2 = do_transform(vec,points_affin)

points3 = do_reverse_transform(vec,points2)
plt.scatter(points3[:,0],points3[:,1],alpha=0.7)




