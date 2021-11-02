import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import cv2 as cv
import PySimpleGUI as sg

classes = ["airplane", "ant", "banana", "baseball", "bird", "bucket", "butterfly", "cat", "coffee cup",
           "dolphin", "donut", "duck", "fish", "leaf", "mountain", "pencil", "smiley face", "snake", "umbrella", "wine bottle"]


class ConvNetwork(nn.Module):
    def __init__(self):
        super(ConvNetwork, self).__init__()
        # self.flatten = nn.Flatten()
        self.conv1 = nn.Conv2d(1, 8, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 20, 5)

        # self.dropout = nn.Dropout()
        self.fc1 = nn.Linear(20 * 4 * 4, 200)
        self.fc2 = nn.Linear(200, 120)
        self.fc3 = nn.Linear(120, 84)
        self.fc4 = nn.Linear(84, 20)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)

        # x = self.dropout(x)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


FILE = 'C:\RUTU\IvLabs\doodle\doodle_model.pth'  # insert path of doodle .pth file on your device
state_dict = torch.load('C:\RUTU\IvLabs\doodle\doodle_model.pth', map_location=torch.device('cpu'))

loaded_model = ConvNetwork()
loaded_model.load_state_dict(state_dict)
loaded_model.eval()

cnn_model = loaded_model

# drawing pad
drawing = False  # true if mouse is pressed
pt1_x, pt1_y = None, None


# mouse callback function
def line_drawing(event, x, y, flags, param):
    global pt1_x, pt1_y, drawing

    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        pt1_x, pt1_y = x, y

    elif event == cv.EVENT_MOUSEMOVE:
        if drawing == True:
            cv.line(img, (pt1_x, pt1_y), (x, y), color=(255, 255, 255), thickness=10)
            pt1_x, pt1_y = x, y
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        cv.line(img, (pt1_x, pt1_y), (x, y), color=(255, 255, 255), thickness=10)


img = np.zeros((420, 420, 1), np.uint8)

cv.namedWindow('test draw')
cv.setMouseCallback('test draw', line_drawing)
sg.theme('DarkTeal6')

layout = [

    [sg.Image(filename='', key='-image-')],
    [sg.Text(size=(80, 1), key='-output-', font='Helvetica 40')],
    [sg.Button('Refresh', size=(7, 1), pad=((600, 0), 3), font='Helvetica 14')],
    [sg.Button('Exit', size=(7, 1), pad=((600, 0), 3), font='Helvetica 14')]]

window = sg.Window('MNIST Digit Classifier', layout, location=(0, 0))

image_elem = window['-image-']

pad = img
test_predict = []

i = 0
while (1):
    event, values = window.read(timeout=0)
    if event in ('Exit', None):
        break
    if event == "Refresh":
        img = np.zeros((420, 420, 1), np.uint8)
    cv.imshow('test draw', img)

    resized_image = cv.resize(img, (28, 28), interpolation=cv.INTER_AREA)
    workingimg = resized_image

    x = torch.from_numpy(workingimg)
    x = x.reshape(1, 1, 28, 28)

    test_out = cnn_model(x.float())
    _, predicted = torch.max(test_out, 1)
    test_predict.append(predicted)

    if len(test_predict) > 2:
        if test_predict[i - 1] != test_predict[i]:
            window["-output-"].update("Looks like " + classes[predicted])
    i += 1

    k = cv.waitKey(1) & 0xFF
    if k == ord('q'):  # press q to quit
        break

    if k == 27:  # escape key to end drawing
        pad = img

        edged = cv.Canny(img, 0, 250)
        (cnts, _) = cv.findContours(edged.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv.boundingRect(cnts[-1])
        img = img[y:y + h, x:x + w]
        resized_image = cv.resize(img, (28, 28), interpolation=cv.INTER_AREA)

        x = torch.from_numpy(resized_image)
        x = x.reshape(1, 1, 28, 28)

        test_out = cnn_model(x.float())
        _, predicted = torch.max(test_out, 1)
        window["-output-"].update("Finally i think its :" + classes[predicted])
        # cv.imshow('Pad', pad)


cv.imshow('Pad', pad)
cv.destroyAllWindows()
