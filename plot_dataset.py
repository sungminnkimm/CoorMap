import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import csv


#f = open("randtestdataset.csv", "w", newline="")
#img = mpimg.imread("testdataset.png")
#timg = mpimg.imread("top.png")


img = mpimg.imread("real.jpg")
timg = mpimg.imread("ue4_topdown.jpg")

plt.close('all')

xdata = []
ydata = []
x2data = []
y2data = []

fig = plt.figure(figsize=(15,15))
ax1, ax2 = fig.subplots(1, 2)
f1, = ax1.plot([], [], 'r.')  # empty point
f2, = ax2.plot([],[],'b.')

def main():

    with open('roomdataset3.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count > 0:
                xdata.append(float(row[0]))
                ydata.append(float(row[1]))
                f1.set_data(xdata, ydata)
                x2data.append(float(row[2]))
                y2data.append(float(row[3]))
                f2.set_data(x2data, y2data)
                line_count = line_count + 1
            else:
                line_count = 1


    ax1.set_title('click to build point segments')
    ax2.set_title('click the corresponding points')

    ax1.imshow(img)
    ax2.imshow(timg)
    plt.show()

if __name__ == '__main__':

    main()











