# Week 4

Welcome to Week 3! This week we will be covering many more powerful classical machine learning algorithms before moving onto modern (aka deep learning) algorithms next week.
So let's get started we have a busy week ahead of us!

## Support Vector Machines
A Support Vector Machine(SVM) is a discriminative classifier formally defined by a separating hyperplane.<br/> 
In other words, given labeled training data(supervised learning), the algorithm outputs an optimal hyperplane which categorizes new examples. In two dimensional space this hyperplane is a line dividing a plane in two parts where in each class lay in either side.
### Python Code:
~~~
#Import Library
from sklearn import svm
#Assumed you have, X(predictor) and y(target) for training data set and x_test(predictor) of test_dataset

#Create SVM object
model = svm.svc()

#Train the model using training set and check score
model.fit(X,y)
model.score(X,y)

#Predict Output
predicted = model.predict(x_test)
~~~

## Naive Bayes
It is a classification technique based on Bayes' theorem with an assumption of idependence between predictors. In simple terms, a Naive Bayes classifier assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature. For example, a fruit may be considered to be an apple if it is red, round, and about 3 inches in diameter.<br/>
Even if these features depend on each other or upon the existence of the other features, a naive bayes classifier would consider all of these properties to independently contribute to the probability that this fruit is an apple. 
Naive Bayesian model is easy to build and particularly useful for very large data sets. Along with simplicity, Naive Bayes is known to outperform even highly sophisticated classification models. 
### Python Code:
~~~
#Import Library
from sklearn.naive_bayes import GaussianNB
#Assumed you have, X(predictor) and y(target) for training data set and x_test(predictor) of test_dataset

#Create Naive Bayes Gaussian object, you could also have chosen bernoulli if required
model = GaussianNB()

#Train the model using training set and check score
model.fit(X,y)
model.score(X,y)

#Predict Output
predicted = model.predict(x_test)
~~~

## k Nearest Neighbours
It can be used for both classification and regression problems. However, it is more widely used in classification problmes in the industry. K nearest neighbors is a simple algorithm that stores all available cases and classifies new cases by a majority vote of its k neighbors. The case being assigned to the class is most common amongst its K nearest neighbors measured by a distance function.<br/>
These distance functions can be Euclidean, Manhattan, Minkowski and Hamming distance. First three functions are used for continuous function and fourth one(Hamming) for categorical variables. If K=1, then the case is simply assigned to the class of its nearest neighbor.<br/>
At times, choosing K turns out to be a challenge while performing kNN modeling.
### Python Code:
~~~
#Import Library
from sklearn.neighbors import KNeighborsClassifier
#Assumed you have, X(predictor) and y(target) for training data set and x_test(predictor) of test_dataset

#Create KNeighbors classifier object
model = KNeighborsClassifier(n_neighbors=6)

#Train the model using training set and check score
model.fit(X,y)
model.score(X,y)

#Predict Output
predicted = model.predict(x_test)
~~~

## Random Forest
Random Forest is a trademark term for an ensemble of decision trees.<br/> 
**Model Ensembling:** Ensemble models are nothing but an aggregation of a number of Weak Learners, a model performing just better than random guessing.<br/>
In Random Forest, we have collection of decision trees(so known as "Forest"). To classify a new object based on attributes, each tree gives a classification and we say the tree "votes" for that class. The forest chooses the classification having the most votes(over all the trees in the forest) 
### Python Code:
~~~
#Import Library
from sklearn.ensemble import RandomForestClassifier
#Assumed you have, X(predictor) and y(target) for training data set and x_test(predictor) of test_dataset

#Create Random Forest object
model = RandomForestClassifier()

#Train the model using training set and check score
model.fit(X,y)
model.score(X,y)

#Predict Output
predicted = model.predict(x_test)
~~~

## Assignment
---
### Kaggle
Kaggle is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

### [Titanic-Machine Learning from Disaster](https://www.kaggle.com/c/titanic/overview)
***
You can download the data from [here](https://www.kaggle.com/c/titanic/data) and participate in the competition.

It's advised to do this by your own but still if you want, you can refer the above file or [this](https://colab.research.google.com/drive/1U57lxLQ2sMIv4j_YPRoCIVw78186iWn6?usp=sharing) link for your pursual. 

The course upto now was devoted to Data Analytics and Machine Learning techniques. So, here we conclude this week and from next week, we'll learn about important Deep Learning techniques where most of research happens presently.
