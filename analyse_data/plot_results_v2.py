import numpy
import numpy as np
import matplotlib.pyplot as plt
import addcopyfighandler

key_photo_index = 0; key_fource_down = 1; key_distance_betwean_centers_of_circles = 2 ;key_part_soap = 3;  key_is_garbage = 4; key_R1 = 5; key_ceter_1_x = 6; key_ceter_1_y = 7 ; key_R2 = 8; key_ceter_2_x = 9; key_ceter_2_y = 10; key_R_std = 11
number_of_fields = 12


def parse_field(v):
    if v == "False":
        return False
    if v == "True":
        return True
    return float(v)

def load_from_file(file_name):
    data_points = []
    with open(file_name) as f:
        lines = f.readlines()
    for line in lines[1:]:
        line = line.replace(", ", ",").replace(" ","")
        fields = [parse_field(v) for v in line.split(",")]
        fields = fields + [np.nan]*(number_of_fields - len(fields))
        data_points.append(fields)
    return np.array(data_points,dtype=object)

def extract_as_data_points(data_original, relable_istrash):
    # split each data row into two data points - comprised of the left and right parts of the string.
    # each data point has only R1 and a1 filled out (opposed to R2 and a2 which are left blank)
    data_points1 = np.copy(data_original)
    data_points2 = np.copy(data_original)
    data_points2[:, key_R1] = data_original[:, key_R2]
    data_points1[:, key_R2] = np.nan
    data_points2[:, key_R2] = np.nan

    # fill in R_error in the key_R_error cells
    # R_uncertainty_std = 0.1
    R_uncertainty_std = (data_original[:,key_R1]-data_original[:,key_R2])/np.sqrt(2)
    data_points1[:,key_R_std] = data_points2[:,key_R_std] = R_uncertainty_std

    # set trashdata flag to be true iff a1 !≈ a2 or 1.25 < a < 0.75
    # partial_differance = (data_original[:,key_a1] - data_original[:,key_a2]) / (data_original[:,key_a1] / 2 + data_original[:,key_a2] / 2)
    # data_points1[:, key_is_garbage] = (abs(partial_differance) > 0.1) + ( 1.25 < data_points1[:, key_a1]) + (data_points1[:, key_a1] < 0.75)
    # data_points2[:, key_is_garbage] = (abs(partial_differance) > 0.1) + ( 1.25 < data_points2[:, key_a1]) + (data_points2[:, key_a1] < 0.75)

    return np.concatenate([data_points1,data_points2])

data_original = load_from_file(r"resaults_analisys_v2")
data_points = extract_as_data_points(data_original,relable_istrash=True)

measured_surface_tensions = []
colors = []

plt.figure()
for part_soap in np.unique(data_points[:,key_part_soap]):
    if part_soap < 0:
        continue

    indexes_with_correct_part_soap = data_points[:,key_part_soap] == part_soap
    idx_soup = indexes_with_correct_part_soap
    idx_garbage = (data_points[:,key_is_garbage] == part_soap)*False
    # idx_garbage = data_points[:,key_R1] > 16

    a = plt.errorbar(
        data_points[idx_soup * (~idx_garbage), key_fource_down], data_points[idx_soup * (~idx_garbage), key_R1],data_points[idx_soup * (~idx_garbage), key_R_std]*2,
        fmt="o",capsize=3,label=f"part_soap={part_soap*100:.3g}%",alpha=0.3)

    colors.append(a.get_children()[0].get_color())

    plt.plot(data_points[idx_soup * (idx_garbage), key_fource_down], data_points[idx_soup * (idx_garbage), key_R1], "xr",alpha=0.2)

    # linear fit
    num_of_points = sum(idx_soup * (~idx_garbage))
    (ST,_),((var_ST,_),(_,_)) = np.polyfit([0] + list(data_points[idx_soup * (~idx_garbage), key_R1]),
               [0] + list(data_points[idx_soup * (~idx_garbage), key_fource_down]),
               deg=1,
               w = [10e6] + [1]*num_of_points,
               cov=True)

    plt.plot([0,900],[0,900/ST],label=f"part_soap={part_soap*100:.3g}: a = {1/ST:3g}",color=colors[-1])

    measured_surface_tensions.append((part_soap,ST,var_ST))

plt.grid(ls="--")
plt.legend()
plt.title("results of curve fitting, each image is a point in this graph\nwith crude 68% confident intervals")
plt.xlabel("string tension [dyne]")
plt.ylabel("string radius [cm]")
plt.xlim(0,850)
plt.ylim(0,16)

plt.figure()
plt.title("68% confidence intervals of the surface tensions\nof the different echo soap concentrations")
plt.xlabel("% soup in solution")
plt.ylabel("surface tension")
measured_surface_tensions = np.array(measured_surface_tensions)
for i in range(len(measured_surface_tensions)):
    plt.errorbar(measured_surface_tensions[i,0]*100,measured_surface_tensions[i,1],measured_surface_tensions[i,2]*2,fmt="none",capsize=3,ecolor=colors[i],label=f"measured for {measured_surface_tensions[i,0]*100:3g}% echo soap")

plt.hlines([30],0,100,colors="green",linestyles="--",label="theory for normal soap")
plt.hlines([72],0,100,colors="red",linestyles="--",label="theory for water")
plt.grid(ls="--")
plt.legend()
plt.xlim(0,3)
plt.ylim(0,90)
