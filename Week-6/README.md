# Welcome to Week 6 of Data Science! 
This week we will be covering the following topics: 
* Introduction to Tensorflow and Keras 
* Introduction to Convolutional Neural Networks
* Practice Assignment - Basic Image Classification in Tensorflow

## Tensorflow and Keras
Last week, you learned about Neural Networks, Backpropagation, Loss Function, Optimizers, Activation functions and how all of these are used together for training a Machine Learning model. But each and every function you defined was written from scratch in Python using classes. Tensorflow is an end-to-end open source tool developed by Google which contains pre-defined modules for all the different loss functions, optimisers, activation functions, layers, data pre-processing tools and much more which can speed up the process of performing machine learning by a huge amount.
<br>Even though all the modules and libraries can be accessed just using Tensorflow, you can also use the **Keras** API with the Tensorflow framework, which also contains pre-trained models like ResNet50, VGG16, etc. in addition to all the functions offered in Tensorflow. For this tutorial, we will be working with Tensorflow keras. 

To get started, first pip install tensorflow and keras and then simply import them in your code. 
~~~ 
pip install tensorflow
pip install keras
import tensorflow as tf
from tensorflow import keras
~~~
Note that you don't need to remember any of the syntax for importing different modules, just google whatever you need. The documentation provided by Tensorflow & Keras is already extremely detailed and easy to understand.<br/> Even an industrial level Data Scientist still depends on it, you can find it [here](https://keras.io/).

### Workflow of ANN
Before coding our first Neural Network using Tensorflow and keras, let us first understand the different phases of deep learning and then, learn how Keras helps in the process of deep learning.
1. Collect required data
- Deep learning requires a lot of input data to successfully learn and predict the result. So, first collect as much data as possible.
2. Analyze data
- Analyze the data and acquire a good understanding of the data. The better understanding of the data is required to select the correct ANN algorithm.
3. Choose an algorithm (model)
- Choose an algorithm, which will best fit for the type of learning process (e.g image classification, text processing, etc) and the available input data. Algorithms are represented by models in Keras. Algorithms include one or more layers. Each layer in ANN can be represented by the Keras Layer in Keras.
4. Prepare data − Process, filter and select only the required information from the data.
5. Split data − Split the data into training and test data sets. Test data will be used to evaluate the prediction of the algorithm / Model (once the machine learns) and to cross check the efficiency of the learning process.
6. Compile the model − Compile the algorithm / model, so that it can be used further to learn by training and finally do prediction. This step requires us to choose loss function and Optimizer. Loss function and Optimizer are used in the learning phase to find the error (deviation from actual output) and do optimization so that the error will be minimized.
7. Fit the model − The actual learning process will be done in this phase using the training data set.
8. Predict result for unknown value − Predict the output for the unknown input data (other than existing training and test data)
9. Evaluate model − Evaluate the model by predicting the output for test data and cross-comparing the prediction with actual result of the test data.
Freeze, Modify or choose new algorithm − Check whether the evaluation of the model is successful. If yes, save the algorithm for future prediction purposes. If not, then modify or choose a new algorithm / model and finally, again train, predict and evaluate the model. Repeat the process until the best algorithm (model) is found.

## Convolutional Neural Networks
### Introduction 
In the previous week, you studied about Neural Networks, specifically Feed Forward Neural Networks, in which you either have one or two hidden layers to map your input to the output. When several hidden layers are used in a Nerual Network, it is known as a **Deep Neural Network (DNN)** and this is where Deep Learning comes into the picture. A DNN is a basic unit of any Deep Learning architecture. 

Till now, you are familiar only with fully connected layers, where each input is mapped to every activation of the next hidden layer with a weight (parameter) associated with it. The total number of parameters between the input layer and the hidden layer thus become equal to the product of elements in both the layers. You are also aware the process of 'Back Propagation' deals with optimising the values of these parameters. Hence, larger the number of parameters of the model, longer the time it will take for training. 

Now let's say you were to do the same process described above, but with image pixel values as your input. Considering even a small image, of say **50x50** pixels, number of input variables will be equal to **2500**. And lets say you are mapping these input pixel values to a fully connected layer with **1000** elements. Obviously, you cannot directly map the input to a smaller feature space, since that will result in a lot of image features not getting learnt by your model, defeating the purpose of using a DNN. So, the number of parameters for this layer is equal to **2500 x 1000 = 2,500,000**. The number of parameters already cross a million and imagine what will happen with more layers. The total number of parameters to be trained will be huge! And remember, this is just for a **50x50** pixel input image. What if you were to use a **1024x1024** sized image? The training time for your model will be even larger this time and hence using fully connected layers is not an efficient approach to deal with image inputs. 
<br>This is where the concept of Convolutional Layers comes in, and a network comprised of convolutional layers (along with fully connected and other layers) is known as a **Convolutional Neural Network** (ConvNet). 
### Convolutional Layers 
Convolutional layers are specifically used in ConvNets when working with images. The use of a Convolutional layer instead of a fully connected layer results in significant decrease in the number of trainable parameters of the model without negligible loss in feature learning. The basic idea of a convolutional layer is to use a filter over the input image, where the filter is a 2D matrix custom designed to learn a particular feature (eg: a vertical edge). The output of the operation is another matrix but of reduced dimensions.
### Pooling Layers
Pooling layers are based on the same concept as convolutional layers, that is, even they perform a filtering operation on the input image matrix, but a pooling filter is not used to detect a particular feature. The aim of a pooling layer is to simply reduce the dimensions of the input matrix to reduce the number of trainable parameters and hence decrease the number of computations, without a huge loss in the features of the input image. Even though there is a trade off between the accuracy of the learned features and computation time, it is very common to make use of pooling layers since the latter is a bigger concern for deep learning models. Pooling layers can be of different types, such as, Max Pooling, Global Average pooling, etc.
### Dropout Layers
Dropout Layer is one of the most popular regularization techniques to reduce overfitting in the deep learning models. Overfitting in the model occurs when it shows more accuracy on the training data but less accuracy on the test data or unseen data.<br/>
In the dropout technique, some of the neurons in hidden or visible layers are dropped or omitted randomly. The experiments show that this dropout technique regularizes the neural network model to produce a robust model which does not overfit.
### Transfer Learning
Sophisticated deep learning models have millions of parameters (weights) and training them from scratch often requires large amounts of data of computing resources. Transfer learning is a technique that shortcuts much of this by taking a piece of a model that has already been trained on a related task and reusing it in a new model.<br/>

For example, if you want to build your own image recognizer that takes advantage of a model that was already trained to recognize 1000s of different kinds of objects within images. You can adapt the existing knowledge in the pre-trained model to detect your own image classes using much less training data than the original model required.<br/>

This is useful for rapidly developing new models as well as customizing models in resource contstrained environments like browsers and mobile devices.

Most often when doing transfer learning, we don't adjust the weights of the original model. Instead we remove the final layer and train a new (often fairly shallow) model on top of the output of the truncated model. 

There are many pre-defined architectures which are widely used for many Deep Learning applications. Some of these are ResNet50, VGG16, Inception and many more. Try to find more information about these architectures and figure out the different layers that are being used in them. This will familiarise you with the ConvNet architectures used in common practice for different applications. 
<br> It is important to note here that these architectures are readily available as modules in the Keras' applications library, so you don't need to build these architectures from scratch! 
## Keras Cheatsheet
As mostly, you will be using Keras for all your projects, it would be more convinient to look at its cheatsheet and refer to it whenever required.   
~~~
#Import
from keras.models import Sequential

from keras.layers import Dense, Activation

from keras.optimizers import SGD


#Sequential model is a linear stack of layers
model = Sequential()


#Stacking layers, first layer needs to receive input size
model.add(Dense(output_dim=64, input_dim=100))

model.add(Activation("tanh"))

model.add(Dense(output_dim=10))

model.add(Activation("softmax"))


#Configure learning process
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True))


#Set epochs and batch size
model.fit(X_train, Y_train, nb_epoch=5, batch_size=32)


#Evaluate performance
loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=32)


#Summarise model
model.summary() # prints summary representation of model

model.get_config() # returns dictionary with config of model

model.get_weights() # returns list of weight tensors


#Summarise layers
layer.get_weights() # returns weights of a layer


#Save model’s architecture, weights, training configurations, state of optimiser to resume training
model.save(filepath) # save entire model

json_string = model.to_json () # save architecture only (JSON or YAML)

model.save_weights(weights.h5) # save weights only (HDF5)


#Load model
keras.models.load_model(filepath) # load entire model

model = model_from_json(json_string) # load architecture model

model.load_weights(weights.h5) # load weights

#Freeze layers
frozen_layer = Dense(32, trainable=False)

#Pretrained models
from keras.applications.vgg16 impoprt VGG16

from keras.applications.vgg19 impoprt VGG19

from keras.applications.resnet50 impoprt ResNet50

from keras.applications.inception_v3 impoprt InceptionV3

model = VGG16(weights='imagenet', include_top=True)
~~~
Here is a fine article on [Convolution networks](https://cs231n.github.io/convolutional-networks/#conv), going through it will be surely beneficial for your visual understanding. 
## Assignment
---
Finally, we will end this week's tutorial by performing an Image Classification task accomplished using Convolutional Neural Networks. For this assignment, you'll design an Image Classification model using CIFAR-10 data set. You don't need to explicitly download the dataset as it's already available in keras datasets collections. Don't forget to use the Keras cheatsheet and follow the steps given in workflow of ANN to build your classification model. 
*** 
The solution to the assignment is given under the Week-6 folder. Please refer to it after trying the assignment by your own.
