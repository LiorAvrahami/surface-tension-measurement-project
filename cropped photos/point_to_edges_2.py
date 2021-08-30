import matplotlib.pyplot as plt
import os
import numpy as np

def to_file_format(points_arr):
    return "\n".join([", ".join(map(str, b)) for b in points_arr])

for fn in sorted(os.listdir()):
    if os.path.splitext(fn)[1] != ".png":
        continue
    data = plt.imread(fn)
    point_f_name = os.path.join(os.path.dirname(fn), os.path.splitext(os.path.split(fn)[1])[0] + "edge_point_data.txt")
    with open(point_f_name) as point_f:
        lines = point_f.readlines()
        points = [[float(v) for v in line.split(",")] for line in lines]
    points = [points[i] if len(points) > i else [np.nan,np.nan] for i in range(4)]
    print("read from file:\n" + to_file_format(points))

    for point_index,point in enumerate(points):
        # if not np.any(np.isnan(point)):
        #     continue

        plt.figure(figsize=(12, 7))
        plt.imshow(data)
        plt.scatter(points[point_index][0],points[point_index][1],c="red")

        if point_index % 2 == 0:
            plt.xlim(plt.xlim()[0], plt.xlim()[1] * 0.3)
        else:
            plt.xlim(plt.xlim()[1] * 0.7, plt.xlim()[1])
        if point_index//2 == 0:
            plt.ylim(plt.ylim()[0] * 0.3, plt.ylim()[1])
        else:
            plt.ylim(plt.ylim()[0]*0.85, plt.ylim()[0] * 0.6)

        def onclick(event):
            if event.button!=3:
                return
            points[point_index] = [event.xdata, event.ydata]
            plt.close()
            print(points[point_index])

        def on_key_press(event):
            if event.key == "escape":
                plt.close()

        plt.gcf().canvas.mpl_connect('button_press_event', onclick)
        plt.gcf().canvas.mpl_connect('key_press_event', on_key_press)
        plt.title(fn)
        plt.show()

    point_f_name = os.path.join(os.path.dirname(fn),os.path.splitext(os.path.split(fn)[1])[0] + "edge_point_data.txt")
    with open(point_f_name,"w+") as out_file:
        str_to_print_to_file = to_file_format(points)
        print("writing to file:\n" + str_to_print_to_file)
        out_file.write(str_to_print_to_file)