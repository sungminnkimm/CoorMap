import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import Cursor
import csv
import numpy as np

#f = open("randtestdataset.csv", "w", newline="")
#img = mpimg.imread("testdataset.png")
#timg = mpimg.imread("top.png")

f = open("test.csv", "w", newline="")
img = mpimg.imread("real.jpg")
timg = mpimg.imread("ue4_topdown.jpg")

writer = csv.writer(f)
plt.close('all')

xdata, ydata= [],[]
x2data, y2data = [],[]
x_scale_data, y_scale_data = [], []
x2_scale_data, y2_scale_data = [], []

writer.writerow(['x1','y1','x2','y2'])
fig = plt.figure(figsize=(10,10))
ax1, ax2 = fig.subplots(1, 2)
f1, = ax1.plot([],[], 'r.')  # empty point
f2, = ax2.plot([],[],'b.')

def add_point(event):
    if event.inaxes!=ax1:
        return

    if event.button == 1:
        x = event.xdata
        y = event.ydata
        x_scale = event.xdata*0.001
        y_scale = event.ydata*0.001
        xdata.append(x)
        ydata.append(y)
        x_scale_data.append(x_scale)
        y_scale_data.append(y_scale)
        f1.set_data(xdata, ydata)
        plt.draw()
        print(x_scale_data, y_scale_data)

    if event.button == 3:
        plt.disconnect(cid)
        print("list1 copied")

def add_point2(event):
    if event.inaxes!=ax2:
        return

    if event.button == 1:

        x2 = event.xdata
        y2 = event.ydata
        x2_scale = event.xdata*0.001
        y2_scale = event.ydata*0.001
        x2data.append(x2)
        y2data.append(y2)
        x2_scale_data.append(x2_scale)
        y2_scale_data.append(y2_scale)
        f2.set_data(x2data, y2data)
        plt.draw()
        print(x2_scale_data, y2_scale_data)

    if event.button == 3:
        plt.disconnect(cid2)
        print("list2 copied")




ax1.set_title('click to build point segments')
ax2.set_title('click the corresponding points')

cid = plt.connect('button_press_event', add_point)
cid2 = plt.connect('button_press_event', add_point2)

cursor = Cursor(ax1, useblit=True, color='white', linewidth=2)
cursor2 = Cursor(ax2, useblit=True, color='white', linewidth=2)

ax1.imshow(img)
ax2.imshow(timg)


plt.show()

mylen = len(xdata)
for i in range(mylen):

    writer.writerow([x_scale_data[i], y_scale_data[i], x2_scale_data[i], y2_scale_data[i]])

f.close()
print("csv file closed")















