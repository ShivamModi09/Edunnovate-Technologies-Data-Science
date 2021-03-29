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
