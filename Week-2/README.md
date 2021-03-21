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

Strictly go through [Linear Regression]() pdf file given above to understand the linear model concepts in detail, as it's a very essential and most basic concept for every complex model in Data Science. 




