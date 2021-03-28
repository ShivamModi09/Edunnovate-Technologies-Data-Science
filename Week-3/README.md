# Week 3
***
Welcome to Week 3! This week we will be covering rest of essential Data Analysis methods which we are indeed a lot helpful and a Data Analyst uses daily in his job. Later next week we'll have a hand on coding experience on a famous real world data: **Kaggle's Titanic - Machine Learning from Disaster** and conclude the field of Data Analytics and start with Deep Learning concepts then.

## Table of content

- Import Data from Module
- Analyzing Individual Feature Patterns using Visualization
- Descriptive Statistical Analysis
- Basics of Grouping
- Correlation and Causation
- ANOVA

## Data Analytics techniques and Assignment of the Week
Use this following assignment link or Download it from above Week 3 files list and give yourself time here as this is final and very important assignment conceptually as well as regarding technical practice wise. You'll learn in detail and apply above given contents in this assignment.
- **[Data Analysis Assignment-1](https://colab.research.google.com/drive/1AdGbO_A40pGOxxBs5kCfvrOyuAQyX5o4?usp=sharing)**
---
Hope you enjoyed and learn atmost you can from the Assignment-1. You must have seen that not only regression solutions are enough for our daily tasks but rather classification tasks play a major role in making our work a lot easier. Majority of advanced algorithms work on classification purpose. So now, let us learn and work on various important Machine Learning algorithms for Classification purpose.

## Logistic Regression
Don't get confused by its name! It is a classification not a regression algorithm.<br/>
It is used to estimate discrete values(Binary values like 0/1,yes/no,true/false) based in given set of independent variable(s). In simple words, it predicts the probability of occurrence of an event by fitting data to a logit function. Hence, it is also known as logit regression. Since, it predicts the probability, its output values lies between 0 and 1.<br/>
The sigmoid function is given by<br/>
y = mx + c <br/>
S(x) = 1/(1+e^(-y)) <br/>
It is S-Shaped and bounded function. It is also called as squashing function, which maps the whole real axis into finite intervals. Usually, the predictions in the classification problem are probability values. So, we don't want our model to predict the probability value to be below 0 or above 1. Sigmoid function helps to achieve that, values below 0.5 is class 0 and values above or equal to 0.5 is class 1. <br/>
### Python Code:
~~~
#Import Library
from sklearn.linear_model import LogisticRegression
#Assumed you have, X(predictor) and y(target) for training data set and x_test(predictor) of test_dataset

#Create logistic regression object
model = LogisticRegression()

#Train the model using training set and check score
model.fit(X,y)
model.score(X,y)

#Equation coefficient and Intercept
print('Coefficient:',model.coef_)
print('Intercept:',model.intercept_)

#Predict Output
predicted = model.predict(x_test)
~~~

## Decision Tree
It is a type of Supervised learning algorithm that is mostly used for classification problems. Suprisingly, it works for both categorical and continuous dependent variables. In this algorithm, we split the population into two or more homogeneous sets. This is done based on most significant attributes/independent variables to make as distinct groups as possible.<br/>
A Decision tree is a flowchart like tree structure, where each internal node denotes a test on an attribute, each branch represents an outcome of the test, and each leaf node (terminal node) holds a class label. Decision trees classify instances by sorting them down the tree from the root to some leaf node, which provides the classification of the instance. An instance is classified by starting at the root node of the tree,testing the attribute specified by this node,then moving down the tree branch corresponding to the value of the attribute and this process is then repeated for the subtree rooted at the new node.<br/>
To split the population into different heterogeneous groups, it uses various techniques like Gini, Information Gain, Chi-square, entropy. 
### Python Code:
~~~
#Import Library
from sklearn import tree
#Assumed you have, X(predictor) and y(target) for training data set and x_test(predictor) of test_dataset

#Create tree object
model = tree.DecisionTreeClassifier(criterion='gini') #here you can change according to your convinience
# model = tree.DecisionTreeRegressor() for regression
#Train the model using training set and check score
model.fit(X,y)
model.score(X,y)

#Predict Output
predicted = model.predict(x_test)
~~~
---
That's all for this week. Next week, we'll encounter other needful machine learning algorithms, till then have fun exercising Data Analytics techniques.  
