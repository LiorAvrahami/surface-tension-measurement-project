import matplotlib.pyplot as plt
import os

for fn in sorted(os.listdir()):
    if os.path.splitext(fn)[1] != ".png":
        continue
    plt.figure(figsize=(12,7))
    data = plt.imread(fn)
    plt.imshow(data)
    xi, xf = plt.xlim()
    yi, yf = plt.ylim()
    def on_resize(some_data):
        global xi, xf,yi, yf
        xi, xf = plt.xlim()
        xi,xf = int(min(xi,xf)),int(max(xi,xf))
        yi, yf = plt.ylim()
        yi,yf = int(min(yi,yf)),int(max(yi,yf))
        print(xi, xf,yi, yf)
    def on_escape(event):
        if event.key == "escape":
            plt.close()
    plt.gcf().canvas.mpl_connect("draw_event",on_resize)
    plt.gcf().canvas.mpl_connect("key_press_event", on_escape)
    plt.title(fn)
    plt.show()
    data = data[yi:yf,xi:xf,:]
    plt.imsave(fn,data)