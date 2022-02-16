import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets  import RectangleSelector

xdata = []
ydata = []

fig, ax = plt.subplots()
line, = ax.plot(xdata, ydata)


def line_select_callback(eclick, erelease):
    x1, y1 = eclick.xdata, eclick.ydata
    # print(x1, y1)
    x2, y2 = erelease.xdata, erelease.ydata
    # print(x2, y2)
    rect = plt.Rectangle( (min(x1,x2),min(y1,y2)), np.abs(x1-x2), np.abs(y1-y2) )
    ax.add_patch(rect)

    geom = np.transpose(rs.geometry)

    for row in geom:
        xdata.append(row[1])
        ydata.append(row[0])

    plt.plot(xdata, ydata)

this = dict(facecolor="red", edgecolor="black", alpha=0.2, fill= False)
rs = RectangleSelector(ax, line_select_callback, drawtype='box', useblit=False, rectprops = this,
                       button=[1], minspanx=5, minspany=5, spancoords='pixels',
                       interactive=True)

plt.grid()
plt.show()