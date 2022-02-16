import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

import matplotlib.patches as patches
from matplotlib.lines import Line2D
mousepress = None
n = 0

class DraggablePolygon:
    lock = None
    def __init__(self):
        print('__init__')
        self.press = None

        fig = plt.figure()
        ax = fig.add_subplot(111)

        self.geometry = []
        self.newGeometry = []
        poly = plt.Polygon(self.geometry, closed=True, fill=False, linewidth=3, color='#F97306')
        ax.add_patch(poly)
        self.poly = poly

    def on_press(event):
        global currently_dragging
        global mousepress
        currently_dragging = True
        if event.button == 3:
            mousepress = "right"
        elif event.button == 1:
            mousepress = "left"

    def on_release(event):
        global current_artist, currently_dragging
        current_artist = None
        currently_dragging = False

    def on_pick(event):
        global current_artist, offset, n
        global listLabelPoints
        if current_artist is None:
            current_artist = event.artist
            if isinstance(event.artist, patches.Circle):
                if event.mouseevent.dblclick:
                    if mousepress == "right":
                        if len(ax.patches) > 2:
                            event.artist.remove()
                            xdata = list(line_object[0].get_xdata())
                            ydata = list(line_object[0].get_ydata())
                            for i in range(0, len(xdata)):
                                if event.artist.get_label() == listLabelPoints[i]:
                                    xdata.pop(i)
                                    ydata.pop(i)
                                    listLabelPoints.pop(i)
                                    break
                            line_object[0].set_data(xdata, ydata)
                            plt.draw()
                else:
                    x0, y0 = current_artist.center
                    x1, y1 = event.mouseevent.xdata, event.mouseevent.ydata
                    offset = [(x0 - x1), (y0 - y1)]
            elif isinstance(event.artist, Line2D):
                if event.mouseevent.dblclick:
                    if mousepress == "left":
                        n = n + 1
                        x, y = event.mouseevent.xdata, event.mouseevent.ydata
                        newPointLabel = "point" + str(n)
                        point_object = patches.Circle([x, y], radius=50, color='r', fill=False, lw=2,
                                                      alpha=point_alpha_default, transform=ax.transData,
                                                      label=newPointLabel)
                        point_object.set_picker(5)
                        ax.add_patch(point_object)
                        xdata = list(line_object[0].get_xdata())
                        ydata = list(line_object[0].get_ydata())
                        pointInserted = False
                        for i in range(0, len(xdata) - 1):
                            if x > min(xdata[i], xdata[i + 1]) and x < max(xdata[i], xdata[i + 1]) and \
                                    y > min(ydata[i], ydata[i + 1]) and y < max(ydata[i], ydata[i + 1]):
                                xdata.insert(i + 1, x)
                                ydata.insert(i + 1, y)
                                listLabelPoints.insert(i + 1, newPointLabel)
                                pointInserted = True
                                break
                        line_object[0].set_data(xdata, ydata)
                        plt.draw()
                        if not pointInserted:
                            print("Error: point not inserted")
                else:
                    xdata = event.artist.get_xdata()
                    ydata = event.artist.get_ydata()
                    x1, y1 = event.mouseevent.xdata, event.mouseevent.ydata
                    offset = xdata[0] - x1, ydata[0] - y1

    def connect(self):
        'connect to all the events we need'
        print('connect')
        self.cidpress = self.poly.figure.canvas.mpl_connect(
        'button_press_event', self.on_press)
        self.cidrelease = self.poly.figure.canvas.mpl_connect(
        'button_release_event', self.on_release)
        self.cidmotion = self.poly.figure.canvas.mpl_connect(
        'motion_notify_event', self.on_motion)

    def on_press(self, event):
        'on button press we will see if the mouse is over us and store some data'
        print('on_press')
        if event.inaxes != self.poly.axes: return
        if DraggablePolygon.lock is not None: return
        contains, attrd = self.poly.contains(event)
        if not contains: return

        if not self.newGeometry:
            x0, y0 = self.geometry[0]
        else:
            x0, y0 = self.newGeometry[0]

        self.press = x0, y0, event.xdata, event.ydata
        DraggablePolygon.lock = self

    def on_motion(self, event):
        'on motion we will move the rect if the mouse is over us'
        if DraggablePolygon.lock is not self:
            return
        if event.inaxes != self.poly.axes: return
        x0, y0, xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress

        xdx = [i+dx for i,_ in self.geometry]
        ydy = [i+dy for _,i in self.geometry]
        self.newGeometry = [[a, b] for a, b in zip(xdx, ydy)]
        self.poly.set_xy(self.newGeometry)
        self.poly.figure.canvas.draw()

    def on_release(self, event):
        'on release we reset the press data'
        print('on_release')
        if DraggablePolygon.lock is not self:
            return

        self.press = None
        DraggablePolygon.lock = None
        self.geometry = self.newGeometry


    def disconnect(self):
        'disconnect all the stored connection ids'
        print('disconnect')
        self.poly.figure.canvas.mpl_disconnect(self.cidpress)
        self.poly.figure.canvas.mpl_disconnect(self.cidrelease)
        self.poly.figure.canvas.mpl_disconnect(self.cidmotion)


dp = DraggablePolygon()
dp.connect()

plt.show()