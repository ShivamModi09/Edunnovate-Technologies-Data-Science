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
SQL, or Structured Query Language, is a language to talk to databases. It allows you to select specific data and to build complex reports. Today, SQL is a universal language of data. It is used in practically all technologies that process data. You can find the SQL cheatsheet under Week-8 folder.
<br/>
<br/>
Till now, we have used Keras for our assignments but there are other frameworks which give certain other benifits over keras. One of the most famous such framework is PyTorch.  
## PyTorch
--- 
## Assignment - Fashion Apparel Recognizer
You will build an Image Classification model for Fashion categories using Fashion-MNIST dataset. The dataset is already available in the framework so you need not download it explicitly. You will use PyTorch framework to build a classfier for this assignment.
***
After trying the problem by your own, You can refer to 'FashionMNISTwithPyTorch.ipynb' file under Week-8 folder.<br/>
With this, we conclude this week as well as this course. Hope you enjoyed it. The most important thing for any technical field like Data Science is practice, so keep practicing. We ensure that doing this course properly will surely build your base conceptually strong. Happy learning!
