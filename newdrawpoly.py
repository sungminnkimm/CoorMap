"""
================
Polygon Selector
================
Shows how one can select indices of a polygon interactively.
"""

import numpy as np

from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path

import matplotlib.image as mpimg
import csv

f = open("smalldataset.csv", 'w', newline="")
writer = csv.writer(f)

class SelectFromCollection:
    """
    Select indices from a matplotlib collection using `PolygonSelector`.
    Selected indices are saved in the `ind` attribute. This tool fades out the
    points that are not part of the selection (i.e., reduces their alpha
    values). If your collection has alpha < 1, this tool will permanently
    alter the alpha values.
    Note that this tool selects collection objects based on their *origins*
    (i.e., `offsets`).
    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        Axes to interact with.
    collection : `matplotlib.collections.Collection` subclass
        Collection you want to select from.
    alpha_other : 0 <= float <= 1
        To highlight a selection, this tool sets all selected points to an
        alpha value of 1 and non-selected points to *alpha_other*.
    """

    # def __init__(self, ax, collection, alpha_other=0.3):
    def __init__(self, ax):
        self.pfig = ax
        self.canvas = ax.figure.canvas
        # self.collection = collection
        # self.alpha_other = alpha_other

        # self.xys = collection.get_offsets()
        # self.Npts = len(self.xys)

        # Ensure that we have separate colors for each object
        # self.fc = collection.get_facecolors()
        # if len(self.fc) == 0:
        #     raise ValueError('Collection must have a facecolor')
        # elif len(self.fc) == 1:
        #     self.fc = np.tile(self.fc, (self.Npts, 1))

        self.poly = PolygonSelector(ax, self.onselect)
        # self.ind = []

    def onselect(self, verts):
        x_list, y_list = [], []
        path = Path(verts)
        vert_list = path.vertices
        print(vert_list)
        for row in vert_list:
            self.pfig.plot(row[0], row[1], 'ro')
            writer.writerow([row[0], row[1]])
        # x_dif = path.vertices[0][0] - initial_vert[0]
        # y_dif = path.vertices[0][1] - initial_vert[1]
        # plt.cla()

        # for row in vert_list:
        #     A = vert_list[0]
        #     B = row
        #     x1, y1 = A[0], A[1]
        #     x2, y2 = B[0], B[1]
        #     midx = (x1+x2)/2
        #     midy = (y1+y2)/2

        # for row in range(len(vert_list)):
        #     A = vert_list[row]
        #     if row == len(vert_list)-1:
        #         B = vert_list[0]
        #     else:
        #         B = vert_list[row+1]
        #     x1, y1 = A[0], A[1]
        #     x2, y2 = B[0], B[1]
        #     stepx, stepy = (x2-x1)/4, (y2-y1)/4
        #     stepx, stepy = (x2 - x1) / 2, (y2 - y1) / 2
        #     tempx = x1
        #     tempy = y1
        #     for i in range(2):
        #         x_list.append(tempx)
        #         y_list.append(tempy)
        #         tempx = tempx + stepx
        #         tempy = tempy + stepy


        # for i in range(4):
        #     x_list.append(tempx)
        #     tempx = tempx + stepx
        #     y_list.append(tempy)
        #     tempy = tempy + stepy
        #     self.pfig.plot(x_list, y_list,'ro')

        # self.ind = np.nonzero(path.contains_points(self.xys))[0]
        # self.fc[:, -1] = self.alpha_other
        # self.fc[self.ind, -1] = 1
        # self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

        # plt.imshow(img)
        # plt.show()

    def disconnect(self):
        self.poly.disconnect_events()
        # self.fc[:, -1] = 1
        # self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    img = mpimg.imread("real.jpg")
    timg = mpimg.imread("ue4_topdown.jpg")



    writer.writerow(['x1','y1','x2','y2'])
    plt.close('all')

    fig, (real_plot, virtual_plot) = plt.subplots(1,2)
    # fig, ax = plt.subplots()
    # f1 = real_plot.plot([],[],'ro')
    # f2 = virtual_plot.plot([], [], 'bo')
    real_plot.imshow(img)
    virtual_plot.imshow(timg)
    # grid_size = 5
    # grid_x = np.tile(np.arange(grid_size), grid_size)
    # grid_y = np.repeat(np.arange(grid_size), grid_size)
    # pts = ax.scatter(grid_x, grid_y)

    # selector = SelectFromCollection(ax, pts)
    selector = SelectFromCollection(real_plot)
    selector2 = SelectFromCollection(virtual_plot)

    print("Select points in the figure by enclosing them within a polygon.")
    print("Press the 'esc' key to start a new polygon.")
    print("Try holding the 'shift' key to move all of the vertices.")
    print("Try holding the 'ctrl' key to move a single vertex.")

    plt.show()

    selector.disconnect()
    selector2.disconnect()

    # After figure is closed print the coordinates of the selected points
    print('\nSelected points:')
    # print(selector.xys[selector.ind])

#############################################################################
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.widgets.PolygonSelector`
#    - `matplotlib.path.Path`