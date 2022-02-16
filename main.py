# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import Cursor

img = mpimg.imread("real.jpg")
timg = mpimg.imread("ue4_topdown.jpg")
# img = mpimg.imread("testdataset.png")
# timg = mpimg.imread("top.png")

# ep = 20000
#
# class Net(nn.Module):
#
#     def __init__(self):
#         super(Net, self).__init__()
#         # 1 input image channel, 6 output channels, 5x5 square convolution
#         # kernel
#         # self.conv1 = nn.Conv2d(1, 6, 5)
#         # self.conv2 = nn.Conv2d(6, 16, 5)
#         # an affine operation: y = Wx + b
#         # self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image
#
#         self.fc1 = nn.Linear(2,4)
#         self.fc2 = nn.Linear(4, 8)
#         self.fc3 = nn.Linear(8, 2)
#         # self.fc4 = nn.Linear(16, 2)
#
#
#     def forward(self, x):
#         # Max pooling over a (2, 2) window
#         # x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
#         # If the size is a square, you can specify with a single number
#         # x = F.max_pool2d(F.relu(self.conv2(x)), 2)
#         # x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
#         # x = F.relu(self.fc1(x))
#         # x = F.sigmoid(self.fc3(x))
#
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         # x = self.fc4(x)
#         return x

ep = 10000

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        # self.conv1 = nn.Conv2d(1, 6, 5)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        # self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image

        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.Linear(8, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 2)
        # self.fc4 = nn.Linear(16, 2)


    def forward(self, x):
        # Max pooling over a (2, 2) window
        # x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        # x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        # x = F.relu(self.fc1(x))
        # x = F.sigmoid(self.fc3(x))

        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = self.fc3(x)
        x = self.fc4(x)
        # x = self.fc4(x)
        return x

def data_loader(train_data_name = "dataset1.csv", split = 0.5):
    data_df = pd.read_csv(train_data_name)
    ll = len(data_df)
    ln = int(ll * split)

    x1_list = data_df['x1'].values[:ln]
    x2_list = data_df['x2'].values[:ln]
    y1_list = data_df['y1'].values[:ln]
    y2_list = data_df['y2'].values[:ln]

    x1_list_test = data_df['x1'].values[ln:]
    x2_list_test = data_df['x2'].values[ln:]
    y1_list_test = data_df['y1'].values[ln:]
    y2_list_test = data_df['y2'].values[ln:]


    return x1_list, x2_list, y1_list, y2_list, x1_list_test, x2_list_test, y1_list_test, y2_list_test


def predict_x2(x1, y1, net):
    return x1+y1

def predict_y2(x1, y1, net):
    return x1+y1


def predict(x1,y1, net):
    input = torch.tensor(np.array([x1,y1], dtype=np.float32))

    res = net(input)
    return res


def checker(x1_list, x2_list, y1_list, y2_list, net, epochs):

    for i in range(len(x1_list)):
        # x2 = predict_x2(x1_list[i], y1_list[i],net)
        # y2 = predict_y2(x1_list[i], y1_list[i],net)
        x2, y2 = predict(x1_list[i], y1_list[i], net)
        print( i, x1_list[i], y1_list[i])
        print( i, x2.item(), y2.item(), x2_list[i], y2_list[i])

    # criterion = nn.MSELoss()
    # test_one = np.concatenate((np.reshape(x1_list,(-1,1)), np.reshape(y1_list,(-1,1))), axis=1)
    # test_one = torch.Tensor(test_one)
    # test_two = np.concatenate((np.reshape(x2_list,(-1,1)), np.reshape(y2_list,(-1,1))), axis=1)
    # test_two = torch.Tensor(test_two)
    #
    # tex = []
    # tly = []
    #
    # if epochs is not None:
    #     for epoch in range(epochs):
    #         out = net(test_one)
    #         loss = criterion(out,test_two)
    #         loss.backward()
    #         tex.append(epoch)
    #         tly.append(loss.item())
    #         print(epoch, loss.item())

    #    plt.plot(tex, tly, 'r', label = 'test')
    #    plt.legend()
    #    plt.xlabel('epoch')
    #    plt.ylabel('loss')


def train(x1_list, x2_list, y1_list, y2_list, net, epochs = 100000):
    optimizer = optim.Adam(net.parameters(), 0.001) ## learning rate
    criterion = nn.MSELoss()


    train_x = np.concatenate((np.reshape(x1_list,(-1,1)), np.reshape(y1_list,(-1,1))), axis=1)
    # print(train_x)
    # print(abcd)
    train_x = torch.Tensor(train_x)

    train_y = np.concatenate((np.reshape(x2_list,(-1,1)), np.reshape(y2_list,(-1,1))), axis=1)

    # print(train_y)

    train_y = torch.Tensor(train_y)

    ex = []
    ly = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        out = net(train_x)
        loss = criterion(out,train_y)
        loss.backward()
        optimizer.step()
        ex.append(epoch)
        ly.append(loss.item())
        print(epoch, loss.item())

    plt.plot(ex, ly, label = 'train')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.axis([0, epochs, 0, 0.0005])




def main():

    plt.close('all')



    x1_list, x2_list, y1_list, y2_list, x1_list_test, x2_list_test, y1_list_test, y2_list_test = data_loader(train_data_name = "smalldataset.csv", split= 1.0)

    net = Net()
    train(x1_list, x2_list, y1_list, y2_list, net, ep)

    checker(x1_list, x2_list, y1_list, y2_list, net, None)

    print("-----------------------------------")

    checker(x1_list_test, x2_list_test, y1_list_test, y2_list_test, net, ep)

    fig, (bx1, bx2) = plt.subplots(1, 2)
    f1, = bx1.plot([], [], 'ro')
    f2, = bx2.plot([], [], 'bo')
    xdata = []
    ydata = []
    x2data = []
    y2data = []

    cursor = Cursor(bx1, useblit=True, color='white', linewidth=2)

    def add_point(event):
        if event.inaxes != bx1:
            return

        if event.button == 1:
            x = event.xdata
            y = event.ydata
            x_scale = event.xdata*0.001
            y_scale = event.ydata*0.001
            xdata.append(x)
            ydata.append(y)
            f1.set_data(xdata, ydata)
            # x2, y2 = predict(x, y, net)
            # x2data.append(x2.item())
            # y2data.append(y2.item())
            x2, y2 = predict(x_scale, y_scale, net)
            x2data.append(x2.item()*1000)
            y2data.append(y2.item()*1000)
            f2.set_data(x2data, y2data)
            plt.draw()

        if event.button == 3:
            print('right click')



    cid = plt.connect('motion_notify_event', add_point)

    bx1.set_title('click for input')
    bx2.set_title('predicted output')


    bx1.imshow(img)
    bx2.imshow(timg)
    plt.show()



# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
