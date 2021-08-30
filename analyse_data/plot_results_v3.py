import numpy
import numpy as np
import matplotlib.pyplot as plt
import addcopyfighandler

key_photo_index = 0; key_fource_down = 1; key_distance_betwean_centers_of_circles = 2 ;key_part_soap = 3;  key_is_garbage = 4; key_R1 = 5; key_ceter_1_x = 6; key_ceter_1_y = 7 ; key_R2 = 8; key_ceter_2_x = 9; key_ceter_2_y = 10; key_R_std = 11
number_of_fields = 12

EXPECTED_GAMMA =0

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

data_points = load_from_file(r"resaults_analisys_v3")

measured_surface_tensions = []
colors = []

plt.figure()
for part_soap in np.unique(data_points[:,key_part_soap]):
    if part_soap < 0:
        continue

    indexes_with_correct_part_soap = data_points[:,key_part_soap] == part_soap
    idx_soup = indexes_with_correct_part_soap
    idx_garbage = data_points[:,key_is_garbage].astype(bool)

    a = plt.scatter(
        data_points[idx_soup * (~idx_garbage), key_fource_down], data_points[idx_soup * (~idx_garbage), key_distance_betwean_centers_of_circles],
        label=f"part_soap={part_soap*100:.3g}%",alpha=0.3)

    colors.append(a.get_edgecolor())

    plt.plot(data_points[idx_soup * (idx_garbage), key_fource_down], data_points[idx_soup * (idx_garbage), key_distance_betwean_centers_of_circles], "xr",alpha=0.2)

    # linear fit
    num_of_points = sum(idx_soup * (~idx_garbage))
    (ST,_),((var_ST,_),(_,_)) = np.polyfit([0] + list(data_points[idx_soup * (~idx_garbage), key_distance_betwean_centers_of_circles]),
               [0] + list(data_points[idx_soup * (~idx_garbage), key_fource_down]),
               deg=1,
               w = [10e6] + [1]*num_of_points,
               cov=True)

    plt.plot([0,2000],[0,2000/ST],label=f"part_soap={part_soap*100:.3g}: ST = {ST:3g}",color=colors[-1])
    std_ST = var_ST**0.5
    measured_surface_tensions.append((part_soap,ST,std_ST))

plt.grid(ls="--")
plt.legend()
plt.title("results of curve fitting, each image is a point in this graph")
plt.xlabel("string mass*gravity [dyne]")
plt.ylabel("string distance between circle centers [cm]")
plt.xlim(0,2000)
plt.ylim(0,40)

plt.figure()
plt.title("68% confidence intervals of the surface tensions\nof the different echo dish soap concentrations")
plt.xlabel("% soup in solution")
plt.ylabel("surface tension [dyn/cm]")
measured_surface_tensions = np.array(measured_surface_tensions)
for i in range(len(measured_surface_tensions)):
    plt.errorbar(measured_surface_tensions[i,0]*100,measured_surface_tensions[i,1],measured_surface_tensions[i,2],fmt="none",capsize=3,linewidth=2,ecolor=colors[i],label=f"measured for {measured_surface_tensions[i,0]*100:.3g}% echo dish soap")

plt.hlines([30],0,100,colors="green",linestyles="--",label="theory for normal soapy water")
plt.hlines([72],0,100,colors="red",linestyles="--",label="theory for water")
plt.grid(ls="--")
plt.legend()
plt.xlim(0,1.5)
plt.ylim(25,75)

plt.show()
