# Week-2 Data Analytics and Machine Learning
***
Welcome to the Week 2! Having laid the foundations of Machine Learning, this week we start exercising Data Analysis. Data Analysis plays an essential role in:
- Discovering Useful Information
- Answering Questions 
- Predicting Future or the unknown

## Pyhton Packages for Data Science
As, Python is used in this course, here are the important Python packages for a data scientist that will be needed for our projects. 

### 1) Scientific Computing Libraries
- Pandas (Data Structures and tools)
- NumPy (Arrays & Matrices)
- SciPy (Integrals, solving Differential Equations, Optimization)

### 2) Visualization Libraries
- Matplotlib (plots & graphs, most popular)
- Seaborn (plots: heat maps, time series, violin plots)

### 3) Algorithmic Libraries
- Scikit-learn (ML: Rergession, Classification)
- Stats-models (Explore data, estimate statistical models and perform statistical tests)

## Machine Learning
As described by Arthur Samuel, “Machine Learning is a science of getting computers to learn without being explicitly programmed”. Tom Mitchell provides a modern definition: "A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in, as measured by P, improves with experience E”.
Example: Email Spam.
E= Watching you label emails spam or not spam.
T= Classifying emails as spam or not.
P= The number(or fraction) of emails correctly classified as spam/not spam.

## Types of Machine Learning Algorithms 
Machine Learning algorithms can broadly classified into two types:
Supervised learning
Unsupervised learning

### 1. Supervised Learning
Supervised learning is a type of machine learning algorithm in which we can predict output from an input on the basis of a given example input-output pairs database. A supervised learning algorithm analyzes the training data and produces an inferred function, which can be used for mapping new examples.

Supervised Learning algorithms can be further categorized into:-
Regression: It is used for predicting a continuous value. For Example, predicting things like to estimate the CO2 emission from a car’s engine or the price of a house based on its characteristics.
Classification: This technique is used for predicting the class or category of a case. For example, if a cell is benign or malignant or filtering spam emails.

### 2. Unsupervised Learning
Unsupervised learning allows us to approach problems with little or no idea what our results should look like. Using unsupervised learning techniques, we are able to explore the structure of our data to extract meaningful information without the guidance of a known outcome variable or reward function.
Example: Given a set of news articles on the web, group them into a set of articles about the same story.

## Linear Regression
Linear Regression is one of the most fundamental models in Machine Learning. It assumes a linear relationship between target variable (y) and predictors (input variables) (x). Formally stating, (y) can be estimated assuming a linear combination of the input variables (x).

When we have a single predictor as the input, the model is called as Simple Linear Regression and when we have multiple predictors, the model is called Multiple Linear Regression.
The input variable (x) is a vector of the features of our dataset, for eg. when we are predicting housing prices from a dataset, the features of the dataset will be Area of the house, Locality, etc. and the output variable (y) in this case will be the housing price. The general representation of a Simple Linear Regression Model is -

                                            y = θ(0) + θ(1)*x
where θ(0) is known as the bias term, and θ in general is called as the weight vector, there are the parameters of the linear regression model. This video gives a detailed explanation of how we arrived at this representation. For multiple linear regression, the equation modifies to -

                                            y = transpose(Θ)*(X) 
where X is a vector containing all the features and Θ is a vector containing all the corresponding weights (bias and weights both) i.e. the parameters of the linear model. Also note that X here has an additonal feature - the constant term 1 which accounts for the intercept of the linear fit.

We define a function called the Cost Function that accounts for the prediction errors of the model. We try to minimize the cost function so that we can obtain a model that fits the data as good as possible. To reach the optima of the cost function, we employ a method that is used in almost all of machine learning called Gradient Descent.

Strictly go through [Linear Regression](https://github.com/ShivamModi09/Edunnovate-Technologies-Data-Science/blob/main/Week-2/LinearRegression.pdf) pdf file given above to understand the linear model concepts in detail, as it's a very essential and most basic concept for every complex model in Data Science. 

Now let us look at us some of other Regression models just for our knowledge but for a note, these are not widely used.  

## Segmented Regression
It is also referred to as Piecewise Regression or Broken Stick Regression. Unlike standard Linear Regression, we fit separate line segments to intervals of independent variables. Formally stating, independent linear models are assigned to individual partitions of feature spaces. This is pretty useful in the cases when the data assumes different linear relationships in separate intervals of the feature values. Generally, applying linear regression on such tasks can incur large statistical losses.

It is important to note that despite it's better accuracy, it can be computationally inefficient because many independent linear regressions have to be performed.

The points where the data is split into intervals are called Break Points. There are various statistical tests that are to be performed in order to determine the break points but the details of those tests are beyond the scope of this course.

## Locally Weighted Regression
In Locally Weighted Regression, we try to fit a line locally to the point of prediction. What it means is that we give more emphasis to the training data points that are close to the point where prediction is to be made. A weight function (generally a bell shaped function) is used to determine the amount of emphasis given to a data point for prediction. This kind of model comes under Non Parametric Learning unlike the Parametric Learning models that we have seen so far.

Such models are useful when the training data is small and has small number of features. It is computationally inefficient because a linear regression is performed everytime we need to predict some value. But at the same time, this allows us to fit even complicated non-linear curves with ease.

## Exploratory Data Analysis
Last week, we learned about Data Visualisation and Exploration. To get hands on experience on Data Analysis on Regression and Classification, refer to the links below -

-[Regression](https://github.com/MicrosoftLearning/Principles-of-Machine-Learning-Python/blob/master/Module2/VisualizingDataForRegression.ipynb)
-[Classification](https://github.com/MicrosoftLearning/Principles-of-Machine-Learning-Python/blob/master/Module2/VisualizingDataForClassification.ipynb)
Now finally let us look into one important aspect of data analysis that is important for machine learning, data cleaning.

## Data Pre-Processing
Let us consider a simple classification problem using logistic regression. Suppose you have 10 columns in your data which would make up the raw features given to you. A naive model would involve training your classifier using all these columns as features. However are all the features equally relevant? This may not be the case. As a worst case example suppose all the data entries in your training set have the same value. Then it does not make sense to consider this as a feature since any tweaking to the parameter corresponding to this feature that you do can be done by changing the bias term as well. This is a redundant input feature that you should remove. Similarly if you have a column that has very low variance it may make sense to remove this feature from your dataset as well. When we work with high dimensional data sometimes it makes sense to work with fewer dimensions. Thus it makes sense to remove the lower variance dimensions. 

Another way we can clean and improve our data is by performing appropriate transformations on the data. Consider the task of Sentiment Classification using Logistic Regression. You are given a tweet and you have to state whether it expresses happy or sad sentiment. You could just take the tweet in and feed it into the classifier (using a particular representation, the details aren't important). But do all the words really matter?

Consider a sample tweet

#FollowFriday @France_Inte @PKuchly57 @Milipol_Paris for being top engaged members in my community this week :)
Clearly any tags in this tweet are irrelevant. Similarly symbols like '#' are also not needed. Thus we need to clean the input data to remove all this unnecesary information. Further in Natural Language Processing words like 'for', 'in' do not contribute to the sentiment and a proper classification would require us to remove this as well. All of this comes under data cleaning and preprocessing.

The preprocessed version of the above tweet would be:

['followfriday', 'top', 'engag', 'member', 'commun', 'week', ':)']
