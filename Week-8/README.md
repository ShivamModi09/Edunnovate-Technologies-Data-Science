# Welcome to the last week of this course! Week 8
This week we'll first learn about other important Data Science topics which need to be known by any Data Scientist. Finally, we'll conclude the week with learning PyTorch and applying it in your Final Assignment. 
As it was introduced earlier that mainly there are 2 types of machine learning algorithms: Supervised and Unsupervised learning. So far, we have discussed many Supervised learning algorithms, so it's time to look at one of most used Unspervised algorithm:- K-Means Clustering. 
## K-Means Clustering
It is a type of unsupervised algorithm which solves the clustering problem. Its procedure follows a simple and easy way to classify a given data set through a certain number of clusters(assume k clusters). Data points inside a cluster are homogeneous and heterogeneous to peer groups.<br/>
How K-means forms cluster:<br/>
1. K-means picks k number of points for each cluster known as centroids.
2. Each data point forms a cluster with the closet centroids i.e, k clusters.
3. Finds the centroids of each cluster based on existing cluster members. Here, we have new centroids.
4. As we have new centroids, repeat step 2 and 3. Find the closest distance for each data point from new centroids and get associated with new k-clusters. Repeat this procedure until convergence occurs i.e, centroids doesn't change.
### Python Code
~~~
#Import Library
from sklearn.cluster import KMeans

#Assumed KNeighbors classifer object model
k_means = KMeans(n_cluster=3,random_state=0)

#Train the model using training set
model.fit(X)

#Predict Output
predicted = model.predict(x_test)
~~~
Now, let us look some important terms of Data Science
## SQL
SQL, or Structured Query Language, is a language to talk to databases. It allows you to select specific data and to build complex reports. Today, SQL is a universal language of data. It is used in practically all technologies that process data. You can find the[ SQL cheatsheet](https://github.com/ShivamModi09/Edunnovate-Technologies-Data-Science/blob/main/Week-8/SQL-cheat-sheet.pdf) under Week-8 folder.
<br/>
<br/>
Till now, we have used Keras for our assignments but there are other frameworks which give certain other benifits over keras. One of the most famous such framework is PyTorch.  
## PyTorch
--- 
Pytorch is a deep learning framework just like Tensorflow, which means: for traditional machine learning models, use another tool for now. PyTorch was developed by Facebook and was first publicly released in 2016. It was created to offer production optimizations similar to TensorFlow while making models easier to write. Because Python programmers found it so natural to use, PyTorch rapidly gained users, inspiring the TensorFlow team to adopt many of PyTorch’s most popular features in TensorFlow 2.0.<br/>
PyTorch has a reputation for being more widely used in research than in production. PyTorch is actually based on Torch, a framework for doing fast computation that is written in C. If you are great with Python and want to be an open source contributor Pytorch is also the way to go.<br/>

#### Comparing PyTorch to Tensorflow, Pytorch is:
- Easier to understand and more pythonic
- Easier to do non-standard or research applications
- Less support from the ecosystem
<br/>
<br/>
It has a very good documentation of its own, you can always refer to it for any kind of difficulties.

## Image Classification Example with PyTorch
One of the popular methods to learn the basics of deep learning is with the MNIST dataset. It is the "Hello World" in deep learning. The dataset contains handwritten numbers from 0 - 9 with the total of 60,000 training samples and 10,000 test samples that are already labeled with the size of 28x28 pixels.
### Step 1) Preprocess the Data
In the first step of this PyTorch classification example, you will load the dataset using torchvision module.
Before you start the training process, you need to understand the data. Torchvision will load the dataset and transform the images with the appropriate requirement for the network such as the shape and normalizing the images.
~~~
import torch
import torchvision
import numpy as np
from torchvision import datasets, models, transforms

# This is used to transform the images to Tensor and normalize it
transform = transforms.Compose(
   [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

training = torchvision.datasets.MNIST(root='./data', train=True,
                                       download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(training, batch_size=4,
                                         shuffle=True, num_workers=2)

testing = torchvision.datasets.MNIST(root='./data', train=False,
                                      download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(testing, batch_size=4,
                                        shuffle=False, num_workers=2)

classes = ('0', '1', '2', '3',
          '4', '5', '6', '7', '8', '9')
         
import matplotlib.pyplot as plt
import numpy as np

#create an iterator for train_loader
# get random training images
data_iterator = iter(train_loader)
images, labels = data_iterator.next()

#plot 4 images to visualize the data
rows = 2
columns = 2
fig=plt.figure()
for i in range(4):
   fig.add_subplot(rows, columns, i+1)
   plt.title(classes[labels[i]])
   img = images[i] / 2 + 0.5     # this is for unnormalize the image
   img = torchvision.transforms.ToPILImage()(img)
   plt.imshow(img)
plt.show()
~~~
The transform function converts the images into tensor and normalizes the value. The function torchvision.transforms.MNIST, will download the dataset (if it's not available) in the directory, set the dataset for training if necessary and do the transformation process. To visualize the dataset, you use the data_iterator to get the next batch of images and labels.
### Step 2) Network Model Configuration
Now in this PyTorch example, you will make a simple neural network for PyTorch image classification.
Here, we introduce you another way to create the Network model in PyTorch. We will use nn.Sequential to make a sequence model instead of making a subclass of nn.Module.
~~~
import torch.nn as nn

# flatten the tensor into 
class Flatten(nn.Module):
   def forward(self, input):
       return input.view(input.size(0), -1)

#sequential based model
seq_model = nn.Sequential(
           nn.Conv2d(1, 10, kernel_size=5),
           nn.MaxPool2d(2),
           nn.ReLU(),
           nn.Dropout2d(),
           nn.Conv2d(10, 20, kernel_size=5),
           nn.MaxPool2d(2),
           nn.ReLU(),
           Flatten(),
           nn.Linear(320, 50),
           nn.ReLU(),
           nn.Linear(50, 10),
           nn.Softmax(),
         )

net = seq_model
print(net)
~~~
Here is the output of our network model
~~~
Sequential(
  (0): Conv2d(1, 10, kernel_size=(5, 5), stride=(1, 1))
  (1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (2): ReLU()
  (3): Dropout2d(p=0.5)
  (4): Conv2d(10, 20, kernel_size=(5, 5), stride=(1, 1))
  (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (6): ReLU()
  (7): Flatten()
  (8): Linear(in_features=320, out_features=50, bias=True)
  (9): ReLU()
  (10): Linear(in_features=50, out_features=10, bias=True)
  (11): Softmax()
)
~~~
### Network Explanation
1. The sequence is that the first layer is a Conv2D layer with an input shape of 1 and output shape of 10 with a kernel size of 5
2. Next, you have a MaxPool2D layer
3. A ReLU activation function
4. a Dropout layer to drop low probability values.
5. Then a second Conv2d with the input shape of 10 from the last layer and the output shape of 20 with a kernel size of 5
6. Next a MaxPool2d layer
7. ReLU activation function.
8. After that, you will flatten the tensor before you feed it into the Linear layer
9. Linear Layer will map our output at the second Linear layer with softmax activation function
### Step 3)Train the Model
Before you start the training process, it is required to set up the criterion and optimizer function.
For the criterion, you will use the CrossEntropyLoss. For the Optimizer, you will use the SGD with a learning rate of 0.001 and a momentum of 0.9 as shown in the below PyTorch example.
~~~
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
~~~
The forward process will take the input shape and pass it to the first conv2d layer. Then from there, it will be feed into the maxpool2d and finally put into the ReLU activation function. The same process will occur in the second conv2d layer. After that, the input will be reshaped into (-1,320) and feed into the fc layer to predict the output.
Now, you will start the training process. You will iterate through our dataset 2 times or with an epoch of 2 and print out the current loss at every 2000 batch.
~~~
for epoch in range(2): 

#set the running loss at each epoch to zero
   running_loss = 0.0
# we will enumerate the train loader with starting index of 0
# for each iteration (i) and the data (tuple of input and labels)
   for i, data in enumerate(train_loader, 0):
       inputs, labels = data

# clear the gradient
       optimizer.zero_grad()

#feed the input and acquire the output from network
       outputs = net(inputs)

#calculating the predicted and the expected loss
       loss = criterion(outputs, labels)

#compute the gradient
       loss.backward()

#update the parameters
       optimizer.step()

       # print statistics
       running_loss += loss.item()
       if i % 1000 == 0:
           print('[%d, %5d] loss: %.3f' %
                 (epoch + 1, i + 1, running_loss / 1000))
           running_loss = 0.0
~~~    
At each epoch, the enumerator will get the next tuple of input and corresponding labels. Before we feed the input to our network model, we need to clear the previous gradient. This is required because after the backward process (backpropagation process), the gradient will be accumulated instead of being replaced. Then, we will calculate the losses from the predicted output from the expected output. After that, we will do a backpropagation to calculate the gradient, and finally, we will update the parameters.
Here's the output of the training process
~~~
[1,  1] loss: 0.002
[1,  1001] loss: 2.302
[1,  2001] loss: 2.295
[1,  3001] loss: 2.204
[1,  4001] loss: 1.930
[1,  5001] loss: 1.791
[1,  6001] loss: 1.756
[1,  7001] loss: 1.744
[1,  8001] loss: 1.696
[1,  9001] loss: 1.650
[1, 10001] loss: 1.640
[1, 11001] loss: 1.631
[1, 12001] loss: 1.631
[1, 13001] loss: 1.624
[1, 14001] loss: 1.616
[2, 	1] loss: 0.001
[2,  1001] loss: 1.604
[2,  2001] loss: 1.607
[2,  3001] loss: 1.602
[2,  4001] loss: 1.596
[2,  5001] loss: 1.608
[2,  6001] loss: 1.589
[2,  7001] loss: 1.610
[2,  8001] loss: 1.596
[2,  9001] loss: 1.598
[2, 10001] loss: 1.603
[2, 11001] loss: 1.596
[2, 12001] loss: 1.587
[2, 13001] loss: 1.596
[2, 14001] loss: 1.603
~~~
### Step 4) Test the Model
After you train our model, you need to test or evaluate with other sets of images. Here, we haven't categorized but in real assignment, you will need to and evaluate using same steps , you did for evaluating training set.

Now, as you have understood how to build an Image Classifier using PyTorch, I suggest you to practice it by your own for your last assignment which is based on famous dataset:- FashionMNIST.

## Assignment - Fashion Apparel Recognizer
You will build an Image Classification model for Fashion categories using Fashion-MNIST dataset. The dataset is already available in the framework so you need not download it explicitly. You will use PyTorch framework to build a classfier for this assignment.
***
After trying the problem by your own, You can refer to 'FashionMNISTwithPyTorch.ipynb' file under Week-8 folder.<br/>
With this, we conclude this week as well as this course. Hope you enjoyed it. The most important thing for any technical field like Data Science is practice, so keep practicing. We ensure that doing this course properly will surely build your base conceptually strong. Happy learning ahead!
