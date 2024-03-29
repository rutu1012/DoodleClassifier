# Doodle Classifier
Draw doodles in real time, and then classify them into one of the chosen classes among 20. Similar to Google's Quick draw we have created a doodle classifier using the concepts of Neural Networks with Deep Learning.

![quickdraw](https://user-images.githubusercontent.com/83291620/137333928-b1415e91-5411-48a7-b296-3c954ee52db3.gif)

# Dataset
The dataset used consisted of 2807037 doodle images of size 28x28 to be classified into 20 classes.   
Dataset link: https://drive.google.com/file/d/18dN1GXMumS_w8o99PDdkXEQUlR1djuSw/view?usp=sharing

# Approach
The **Prelimnary Stage** involved studying and learning the basics of Machine Learning and Deep Learning algorithms .

For better understanding of the topic, developed a **Digit Classifier** from scratch using the MNIST dataset coded using numpy. The writing of all the functions from scratch for the Forward and Backward propagation along with activations and calculating gradients and putting it all into an iterative learning function helped strengthen the concepts.

The **CNN model** is coded with the help of pytorch library for the convolution of image with filters along with maxpooling. After multiple convolutional layers, the input representation is flattened into a feature vector and passed through a network of neurons to predict the output probabilities.

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
Epochs   | 25
Activation  | ReLU
Loss      | Cross Entropy Loss
Optimizer  | Adams
Betas   | (0.9, 0.999)

# Results
Loss vs No. of Epochs graph  
![loss graph](https://user-images.githubusercontent.com/83291620/149280732-409b340b-c512-48e9-9cec-4f82313c0cff.png)



### Accuracy ###

Dataset   | Accuracy
------- | -------------
Train set   | 92.122%
Test set  |  90.635%

# Final Outcome
![doodle_gif_f](https://user-images.githubusercontent.com/83291620/139815477-20dd85e7-a673-48a6-bbf3-7e1b7c66d561.gif)


# Resources referred
- [Coursera Deep Learning](https://www.coursera.org/specializations/deep-learning)
- [Pytorch documentation](https://pytorch.org/docs/stable/index.html)

