# Week 5
***
Welcome to Week 5! This week you are going to develop your understanding about neural networks and deep learning. From here we'll start learning about one of the basic and crucial part of Data Scientist job. This week we'll learn a lot if interesting theory so if you concentrate, we ensure you'll surely have fun.

Last week you studied about various machine learning algorithms.It’s a pertinent question. There is no shortage of machine learning algorithms so why should we gravitate towards deep learning algorithms? What do neural networks offer that traditional machine learning algorithms don’t?

## Why are neural nets preferred over traditional machine learning algorithms ?
- **Decision Boundary**: In the case of classification problems, the algorithm learns the function that separates 2 classes, this is known as a Decision boundary.
**_Every Machine Learning algorithm is not capable of learning all the functions. This limits the problems these algorithms can solve that involve a complex relationship whereas Neural Nets can compute any function._**
* **Feature Engineering:** Feature engineering is a key step in the model building process. It is a two-step process:
   * Feature extraction: In feature extraction, we extract all the required features for our problem statement
   * Feature selection: In feature selection, we select the important features that improve the performance of our machine learning or deep learning model.<br />

First of All, let us understand what Neural Networks actually are and where this idea came from.

## What are Neural Networks ?

Human Brain is the most wonderful entity and it’s really amazing to understand how it works. Our brain has the ability to learn everything itself on the basis of its own learning algorithm which led to an idea of imitating the human brain and creating Neural networks whose working principles are based on the human brain.

Recently in the field of computer science neural networks has attracted a great deal of attention from many people. From doing various tasks like face recognition and speech recognition, to Healthcare and Marketing, Neural networks are widely used today. 

An artificial Neural network is the functional unit of Deep learning which is itself a part of Machine learning which itself is a subfield of Artificial Intelligence. 
Neural networks were developed as simulating neurons or networks of neurons in the brain. Neurons have input wires called Dendrites which receive information from various locations. A neuron also has an output wire called Axon which sends messages to other neurons. Similarly an artificial Neural networks function when an input data is provided to it. Then this data is processed via layers of perceptrons to give a desired output. 

Let’s try to understand this with an example. Consider a situation in which the task is to recognize the given shape. Let consider the shape to be distributed into 28\*28 pixels which makes up for 784 pixels and is fed as an input to each neuron of the first layer. Neurons of the first layer are connected to neurons of the next layer through channels and each of these channels has its own value known as Weight. This value is then passed through a threshold function called the Activation function. The result of the activation function determines if the particular neuron is activated or not.

An activated neuron transmits data to the neurons of the next layer over the channels and this is how the data is propagated through the network and this is called Forward propagation. In the output layer the neuron with the highest value determines the output. These values are basically a probability.

Our network also has the output fed to it. The predicted output is then compared with the actual output to calculate the error in the prediction. The magnitude of the error indicates how wrong we are and the sign suggests if our predicted values are higher or lower than expected. This information is then relocated backward through our network. This is known as Back propagation. Based on this information the weights are adjusted. This cycle of forward propagation and backpropagation is iteratively performed with multiple inputs. The process goes on until we get appropriate weights that produce correct output and predict our shape correctly.

BackPropagation: The main feature of backpropagation is its iterative, recursive and efficient method for calculating the weights updates to improve the network until it is able to perform the task for which it is being trained.
This was a basic idea of how Artificial Neural Networks actually work.

You can find whole our article paper on this in above Week-5 folder's Neural Network pdf. Strictly Go through that as it's important to understand it deeply.

Now, it's obvious to think that what's Deep Learning and how's it different from Neural Networks, so let us clarify it as it's important to keep that in mind. 

## What is the difference between Deep Learning and Neural Networks?
The obvious difference between Deep Learning and Neural Networks can be that Neural Networks operates similar to neurons in the human brain to perform various computation tasks faster while Deep Learning is a special type of machine learning that imitates the learning approach humans use to gain knowledge.

## Types of Neural Networks
Different types of neural networks are used for different data and applications. The different architectures of neural networks are specifically designed to work on those particular types of data or domain. Let’s start from the most basic ones and go towards more complex ones.

**Perceptron**
The Perceptron is the most basic and oldest form of neural networks. It consists of just 1 neuron which takes the input and applies activation function on it to produce a binary output. It doesn’t contain any hidden layers and can only be used for binary classification tasks.

The neuron does the processing of addition of input values with their weights. The resulted sum is then passed to the activation function to produce a binary output.
**Feed Forward Network**
The Feed Forward (FF) networks consist of multiple neurons and hidden layers which are connected to each other. These are called “feed-forward” because the data flow in the forward direction only, and there is no backward propagation. Hidden layers might not be necessarily present in the network depending upon the application.
More the number of layers more can be the customization of the weights. And hence, more will be the ability of the network to learn. Weights are not updated as there is no backpropagation. The output of multiplication of weights with the inputs is fed to the activation function which acts as a threshold value.
FF networks are used in:
- Classification
- Speech recognition
- Face recognition
- Pattern recognition

**Multi-Layer Perceptron**
The main shortcoming of the Feed Forward networks was its inability to learn with backpropagation. Multi-layer Perceptrons are the neural networks which incorporate multiple hidden layers and activation functions. The learning takes place in a Supervised manner where the weights are updated by the means of Gradient Descent.

Multi-layer Perceptron is bi-directional, i.e., Forward propagation of the inputs, and the backward propagation of the weight updates. The activation functions can be changes with respect to the type of target. Softmax is usually used for multi-class classification, Sigmoid for binary classification and so on. These are also called dense networks because all the neurons in a layer are connected to all the neurons in the next layer.

They are used in Deep Learning based applications but are generally slow due to their complex structure.

**Radial Basis Networks**
Radial Basis Networks (RBN) use a completely different way to predict the targets. It consists of an input layer, a layer with RBF neurons and an output. The RBF neurons store the actual classes for each of the training data instances. The RBN are different from the usual Multilayer perceptron because of the Radial Function used as an activation function.

When the new data is fed into the neural network, the RBF neurons compare the Euclidian distance of the feature values with the actual classes stored in the neurons. This is similar to finding which cluster to does the particular instance belong. The class where the distance is minimum is assigned as the predicted class.

The RBNs are used mostly in function approximation applications like Power Restoration systems.

**Convolutional Neural Networks**
When it comes to image classification, the most used neural networks are Convolution Neural Networks (CNN). CNN contain multiple convolution layers which are responsible for the extraction of important features from the image. The earlier layers are responsible for low-level details and the later layers are responsible for more high-level features.

The Convolution operation uses a custom matrix, also called as filters, to convolute over the input image and produce maps. These filters are initialized randomly and then are updated via backpropagation. One example of such a filter is the Canny Edge Detector, which is used to find the edges in any image. 

After the convolution layer, there is a pooling layer which is responsible for the aggregation of the maps produced from the convolutional layer. It can be Max Pooling, Min Pooling, etc. For regularization, CNNs also include an option for adding dropout layers which drop or make certain neurons inactive to reduce overfitting and quicker convergence.

CNNs use ReLU (Rectified Linear Unit) as activation functions in the hidden layers. As the last layer, the CNNs have a fully connected dense layer and the activation function mostly as Softmax for classification, and mostly ReLU for regression.

**Recurrent Neural Networks**
Recurrent Neural Networks come into picture when there’s a need for predictions using sequential data. Sequential data can be a sequence of images, words, etc. The RNN have a similar structure to that of a Feed-Forward Network, except that the layers also receive a time-delayed input of the previous instance prediction. This instance prediction is stored in the RNN cell which is a second input for every prediction.

However, the main disadvantage of RNN is the Vanishing Gradient problem which makes it very difficult to remember earlier layers’ weights.

**Long Short-Term Memory Networks**
LSTM neural networks overcome the issue of Vanishing Gradient in RNNs by adding a special memory cell that can store information for long periods of time. LSTM uses gates to define which output should be used or forgotten. It uses 3 gates: Input gate, Output gate and a Forget gate. The Input gate controls what all data should be kept in memory. The Output gate controls the data given to the next layer and the forget gate controls when to dump/forget the data not required. 

LSTMs are used in various applications such as:

- Gesture recognition
- Speech recognition
- Text prediction

But, the Networks we'll work on are explained in detail in the 'Types of Neural Networks' pdf under week-5 section.

## **Forward Propagation**: <br/>
To put it simply, the process that runs inside a neuron for forward propagation is, the neuron takes n input data (training examples) x1, x2,..., xn and first assigns random weights and biases to all the input variables and calculate their weighted sum Z, and further pass it inside an activation function, g(x) such as a sigmoid (later we'll also talk about other alternatives for it),giving g(Z) = A. 
<br/>
Similarly, we calculate this activation, A for all the neurons in all the layers. The activations of the previous layer act as input data for the next layer and the activation of the last layer gives the output, y^(y-hat)
<br/>
### **Activation Functions** <br/>
We can use different types of activation functions such as sigmoid, tanh, Relu (rectified linear unit), leaky relu.
Though, we generally prefer tanh over sigmoid since, both have similar properties, but tanh gives output whose mean can be centralised to 0 and has some other benefits too.
Both of these activation functions, share one disadvantage that the slope tends to 0 when the numbers are very large or very small, thus the process of making improvememts slows down here, the leaky Relu function serves as a benefit in this case as it's slope never collapses. However, for practical purposes even relu function works well. While using for classification, we prefer the output of the last layer to be between 0 and 1, thus we use L-1 layers with relu activation and the last year with sigmoid activation for good results in a L layer NN.<br/>
By now, our model has made it's first prediction, but this was with random weights and biases, hence the results were very random, we need to train our model. Now, to begin it's training the model must know where it went wrong. thus it needs a function to compute loss.

### **Optimization Problem**<br/>
Typically, a neural network model is trained using the gradient descent optimization algorithm and weights are updated using the backpropagation of error algorithm.The gradient descent algorithm seeks to change the weights so that the next evaluation reduces the error, meaning the optimization algorithm is navigating down the gradient (or slope) of error.Now that we know that training neural nets solves an optimization problem, we can look at how the error of a given set of weights is calculated.<br/>

## **Loss Function**<br/>
With neural networks, we seek to minimize the error. As such, the objective function is often referred to as a cost function or a loss function and the value calculated by the loss function is referred to as simply “loss.”

### **Maximum Likelihood And Cross Entropy** <br/>
Maximum likelihood seeks to find the optimum values for the parameters by maximizing a likelihood function derived from the training data.<br/>
When modeling a classification problem where we are interested in mapping input variables to a class label, we can model the problem as predicting the probability of an example belonging to each class. In a binary classification problem, there would be two classes, so we may predict the probability of the example belonging to the first class.<br/>
Therefore, under maximum likelihood estimation, we would seek a set of model weights that minimize the difference between the model’s predicted probability distribution given the dataset and the distribution of probabilities in the training dataset. This is called the cross-entropy. <br/> <br/>

**Cross Entropy Loss and Mean Squared Loss(MSE)** are the losses we use of classification and regression problems respectively. 

## Back Propagation

Under Back Propagation, we use the loss calculated using loss function to make currections to our model. For this, we use Gradient Descent and then update our parameters accordingly. We keep on repeating forward and backward propagation for many epochs to decrease value of cost function and increase accuracy.

## Multi-class Classification using NN
Rather than just to classify an object between yes/no. If we wish to classify it into more than 2 items using NN, we can do it similarly just the only difference will be that the output yhat will be a Mx1 matrix rather than 1x1 for a m class classifier.

- [Multi Layer NN](http://ufldl.stanford.edu/tutorial/supervised/MultiLayerNeuralNetworks/) is a worth to compile your understandings
- Check this [link](https://medium.com/@shaistha24/basic-concepts-you-should-know-before-starting-with-the-neural-networks-nn-3-6db79028e56d) for an intuitive run through and answers to few questions rising inside you.

## Assignment
***
Implementing all of this in a code and making a neural network of your own will make your understanding better. <br/>
Use Vectorization to do so rather than using for loops. 

**Important:** Do not use scikit learn or keras or any other libraries. Implement the codes from scratch using numpy.<br/>
Implement seperate functions such as initialization, forward propagation, cost calculation and back propagation and then compile all of it in a class/function and test your neural net.<br/>

**Additional Resource:** If you don't know how to do so, you can refer to [this](https://towardsdatascience.com/vectorization-implementation-in-machine-learning-ca652920c55d) article.

---

That's it for this week. Next week you'll learn tensorflow and pytorch which will work as very helpful tools to implement all the algorithms you would have learnt upto then. Next week's going to last of fun learning, Stay Tuned.
