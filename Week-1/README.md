# Week-1 Brushing Up the Basics
***
Welcome to the course! This week, the topics we'll discuss about-

- Introduction
- Data Visualisation using Matplotlib
- Data Distribution
- Data Preparation using Pandas
- Numpy
- Mathematics for Data Science

## Introduction

Even though Artificial Intelligence doesn't actually need an introduction in today's world, where millions of people research in this field, and where every other day there's a new state of the art techinique. Data Science is basically automating and improving the learning process of computers based on their experiences without being actually programmed i.e. without any human assistance. At its core, data science is a field of study that aims to use a scientific approach to extract meaning and insights from data. Machine learning, on the other hand, refers to a group of techniques used by data scientists that allow computers to learn from data. These techniques produce results that perform well without programming explicit rules.

In Traditional Programming, We feed in Data (Input) + Program (Logic), run it on machine and get output.

While in Machine Learning, We feed in Data (Input) + Output, run it on machine during training and the machine creates its own program(Logic), which can be evaluated while testing.

Excited? Now go on, begin your journey into this vast and the most buzzing field in Computer Science here.

## Data Visualization using Matplotlib

Before creating analytical models, a data scientist must develop an understanding of the properties and relationships in a dataset. There are two goals for data exploration and visualization -
- To understand the relationships between the data columns.
- To identify features that may be useful for predicting labels in machine learning projects. Additionally, redundant, collinear features can be identified.

Thus, visualization for data exploration is an essential data science skill. Here, we’ll learn to analyze data via various types of plots offered by matplotlib and seaborn library.

### Introduction to Matplotlib
Matplotlib is an amazing visualization library in Python for 2D plots of arrays. Matplotlib is a multi-platform data visualization library built on NumPy arrays and designed to work with the broader SciPy stack. It was introduced by John Hunter in the year 2002.
One of the greatest benefits of visualization is that it allows us visual access to huge amounts of data in easily digestible visuals. Matplotlib consists of several plots like line, bar, scatter, histogram etc.

Follow [this](https://colab.research.google.com/drive/1UoTT78DlYcOcDCYrWNC9pX_x8AWbR40C?usp=sharing) matplotlib tutorial to get hands on coding experience.

## Data Distribution

The data that we have for our model can come from a variety of distributions. Having a sound statistical background can be greatly beneficial in the daily life of a Data Scientist. Every time we start exploring a new dataset, we need to first do an Exploratory Data Analysis (EDA) in order to get a feeling of what are the main characteristics of certain features. If we are able to understand if it’s present any pattern in the data distribution, we can then tailor-made our Machine Learning models to best fit our case study. In this way, we will be able to get a better result in less time (reducing the optimisation steps). In fact, some Machine Learning models are designed to work best under some distribution assumptions. Therefore, knowing with which distributions we are working with, can help us to identify which models are best to use. Let us briefly talk about some data distributions -

### Bernoulli Distribution
The Bernoulli distribution is one of the easiest distributions to understand and can be used as a starting point to derive more complex distributions.
This distribution has only two possible outcomes and a single trial.
A simple example can be a single toss of a biased/unbiased coin. In this example, the probability that the outcome might be heads can be considered equal to p and (1 - p) for tails (the probabilities of mutually exclusive events that encompass all possible outcomes needs to sum up to one). Go through [this](https://towardsdatascience.com/understanding-bernoulli-and-binomial-distributions-a1eef4e0da8f) article to gain deeper understanding.

### Uniform Distribution
The Uniform Distribution can be easily derived from the Bernoulli Distribution. In this case, a possibly unlimited number of outcomes are allowed and all the events hold the same probability to take place.
As an example, imagine the roll of a fair dice. In this case, there are multiple possible events with each of them having the same probability to happen. Go through [this](https://www.probabilitycourse.com/chapter4/4_2_1_uniform.php) article to gain deeper understanding.

### Normal (Gaussian) Distribution
The Normal Distribution is one of the most used distributions in Data Science. Many common phenomena that take place in our daily life follows Normal Distributions such as: the income distribution in the economy, students average reports, the average height in populations, etc… In addition to this, the sum of small random variables also turns out to usually follow a normal distribution (Central Limit Theorem).
Some of the characteristics which can help us to recognise a normal distribution are:
The curve is symmetric at the centre. Therefore mean, mode and median are all equal to the same value, making distribute all the values symmetrically around the mean.
The area under the distribution curve is equal to 1 (all the probabilities must sum up to 1). Go through [this](https://www.mathsisfun.com/data/standard-normal-distribution.html) article to gain deeper understanding.

## Data Preparations using Pandas

### Introduction to Pandas
It's an open source data analysis library for providing easy-to-use data structures and data analysis tools. It comes handy for data manipulation and analysis.

### Benifits
- A lot of functionality
- Active community
- Extensive documentation
- Plays well with other packages
- Built on top of NumPy
- Works well with scikit-learn

To gain hands on experience, check out [this](https://github.com/MicrosoftLearning/Principles-of-Machine-Learning-Python/blob/master/Module3/DataPreparation.ipynb) assignment.

## Numpy 
### Introduction
NumPy (Numerical Python) is a linear algebra library in Python. It is a very important library on which almost every data science or machine learning Python package such as SciPy (Scientific Python), Mat−plotlib (plotting library), Scikit-learn, etc depends on to a great extent. 
NumPy is very useful for performing mathematical and logical operations on Arrays. It provides an abundance of useful features for operations on n-arrays and matrices in Python. 
One of the main advantages of Numpy is that vectorisation using numpy arrays makes it super time efiicient. It enables parallel computation that makes it so fast and hence extremely useful.

## Mathematics for Data Science

Calculus and Linear Algebra are an integral part of Machine Learning. Let us brush up the basics so that our study of Machine Learning can be smooth and rigorous. For coding, we'll be using data science libraries which makes our work a lot easier but for theory and research purposes, one should have a lot grasp on them as these are very much needful.   

### Calculas
Read [this](https://towardsdatascience.com/calculus-in-data-science-and-its-uses-3f3e1b5e5b35) thoroughly and understand the concepts well so that it becomes easy for you to explore the field well.
### Linear Algebra
Read [this](https://towardsdatascience.com/boost-your-data-sciences-skills-learn-linear-algebra-2c30fdd008cf) to get a hang of the concepts that are employed in Machine Learning.

Don’t worry if you don’t understand all of it at once. You can skip this for now and maybe read it later as per your need.
