# Doodle Classifier
Draw doodles in real time, and then classify them into one of the chosen classes among 20. Similar to Google's Quick draw we have created a doodle classifier using the concepts of Neural Networks with Deep Learning.

![quickdraw](https://user-images.githubusercontent.com/83291620/137333928-b1415e91-5411-48a7-b296-3c954ee52db3.gif)

# Dataset
The dataset used consisted of 2807037 doodle images of size 28x28 to be classified into 20 classes.   
Dataset link: https://drive.google.com/drive/u/0/folders/1Sr44plRzk8xRGrWUwtuHQc8XNkJpAN6U

# Approach
The **Prelimnary Stage** involved studying and learning the basics of Machine Learning and Deep Learning algorithms .

For better understanding of the topic, developed a **Digit Classifier** from scratch using the MNIST dataset coded using numpy. The writing of all the functions from scratch for the Forward and Backward propagation along with activations and calculating gradients and putting it all into an iterative learning function helped strengthen the concepts.

Coding the **CNN model** of the network by using torch for convolution of image with filters along with maxpooling. After multiple convolutional layers, the input representation is flattened into a feature vector and passed through a network of neurons to predict the output probabilities.

A **Drawing Pad** is also created using OpenCv to facilitate the user in giving inputs in drawing the doodle for the developed model to classify it.

# CNN Architecture
Training a Convolutional Neural Network with 2 **Convolutional Layers**, both using a filter size of 5x5:

Layer  | Kernel size   | No. of Filters
------ | ------------- | -------------
Conv1  |    (5,5)      |      8
Conv2  |    (5,5)      |      20

**Fully Connected Layer**

Layer   | Size
------- | -------------
fcc 1   | (20 x 4 x 4, 200)
fcc 2   | (200, 120)
fcc 3   | (120, 84)
fcc 4   | (84, 20)

### Hyperparamters ###

Parameters   | Value
------------ | -------------
Learning rate  | 0.001
Mini batch size   | 512
Epochs   | 70
Activation  | ReLU
Loss      | Cross Entropy Loss
Optimization  | Adams
Betas   | (0.9, 0.999)
