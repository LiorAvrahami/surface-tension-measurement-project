import matplotlib.pyplot as plt
import os
import addcopyfighandler

for fn in sorted(os.listdir()):
    if os.path.splitext(fn)[1] != ".png":
        continue

    data = plt.imread(fn)
    point_f_name = os.path.join(os.path.dirname(fn), os.path.splitext(os.path.split(fn)[1])[0] + "edge_point_data.txt")
    with open(point_f_name) as point_f:
        lines = point_f.readlines()
        (x0,y0),(x1,y1) = [[float(v) for v in line.split(",")] for line in lines]

    plt.figure(figsize=(12,7))

    plt.imshow(data)
    plt.plot([x0,x1],[y0,y1],"r--o")

    def on_escape(event):
        if event.key == "escape":
            plt.close()
    plt.gcf().canvas.mpl_connect("key_press_event", on_escape)
    plt.title(fn)
    plt.show()