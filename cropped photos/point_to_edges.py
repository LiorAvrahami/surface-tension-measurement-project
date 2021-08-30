import matplotlib.pyplot as plt
import os



for fn in sorted(os.listdir()):
    if os.path.splitext(fn)[1] != ".png":
        continue
    data = plt.imread(fn)

    plt.figure(figsize=(12, 7))
    plt.imshow(data)
    plt.xlim(plt.xlim()[0], plt.xlim()[1] * 0.3)
    plt.ylim(plt.ylim()[0]*0.3, plt.ylim()[1])
    def onclick(event):
        if event.button!=3:
            return
        global x0,y0
        x0,y0 = event.xdata, event.ydata
        plt.close()
        print(x0,y0)
    plt.gcf().canvas.mpl_connect('button_press_event', onclick)
    plt.title(fn)
    plt.show()

    plt.figure(figsize=(12, 7))
    plt.imshow(data)
    plt.xlim(plt.xlim()[1]*0.7, plt.xlim()[1])
    plt.ylim(plt.ylim()[0]*0.3, plt.ylim()[1])
    def onclick(event):
        if event.button!=3:
            return
        global x1,y1
        x1,y1 = event.xdata, event.ydata
        plt.close()
        print(x1,y1)
    plt.gcf().canvas.mpl_connect('button_press_event', onclick)
    plt.title(fn)
    plt.show()

    point_f_name = os.path.join(os.path.dirname(fn),os.path.splitext(os.path.split(fn)[1])[0] + "edge_point_data.txt")
    with open(point_f_name,"w+") as out_file:
        out_file.write(f"{x0},{y0}\n{x1},{y1}")