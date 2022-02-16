import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets  import PolygonSelector
import matplotlib.image as mpimg

xdata = []
ydata = []
img = mpimg.imread("real.jpg")
timg = mpimg.imread("ue4_topdown.jpg")
fig, (ax,ax2) = plt.subplots(1,2)
#
#
# def line_select_callback(eclick, erelease):
#     x1, y1 = eclick.xdata, eclick.ydata
#     x2, y2 = erelease.xdata, erelease.ydata
#
#     poly = plt.Polygon( (min(x1,x2),min(y1,y2)), np.abs(x1-x2), np.abs(y1-y2) )
#     ax.add_patch(poly)
#
#     geom = np.transpose(ps.verts)
#     print(ps.verts)
#
#     for row in geom:
#         xdata.append(row[1])
#         ydata.append(row[0])
#
#
# ps = PolygonSelector(ax, line_select_callback, useblit=False, vertex_select_radius=15)
# ps2 = PolygonSelector(ax2, line_select_callback, useblit=False, vertex_select_radius=15)
#

def line_select_callback(eclick):
    x1, y1 = eclick.xdata, eclick.ydata
    # x2, y2 = erelease.xdata, erelease.ydata

    poly = plt.Polygon( (min(x1),min(y1)), np.abs(x1), np.abs(y1) )
    ax.add_patch(poly)

    geom = np.transpose(ps.verts)
    print(ps.verts)

    for row in geom:
        xdata.append(row[1])
        ydata.append(row[0])


ps = PolygonSelector(ax, line_select_callback, useblit=False, vertex_select_radius=15)
ps2 = PolygonSelector(ax2, line_select_callback, useblit=False, vertex_select_radius=15)




ax.imshow(img)
ax2.imshow(timg)
plt.show()