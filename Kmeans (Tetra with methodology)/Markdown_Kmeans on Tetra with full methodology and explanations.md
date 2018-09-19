

```python

```


```python
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random


% matplotlib inline
```


```python
# The goal is to implement kmeans clustering which is based on Lloyd's algorithm
# a :We start by placing centroids randomly on the plane
# b : Then we assign each point to the centroid closest to it
# c :Then we reassign the centroid to become the center (mean) of that cluster
# d: repeat steps a thru c until the centroids arent moving
```


```python
# We build the python code with a simple dataset and then test it on the tetra data
```


```python
# First we have to import the tetra dataset using pandas
```


```python
# The set starts at the 4th row
```


```python
tetra = pd.read_table('Tetra.lrn', skiprows = 3, usecols= [1,2,3] )
```


```python
#Eyeball the data to make sure it loaded
```


```python
tetra[:5] 
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>C1</th>
      <th>C2</th>
      <th>C3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.295428</td>
      <td>0.050829</td>
      <td>-0.385217</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.409178</td>
      <td>-0.035191</td>
      <td>-0.251980</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.096803</td>
      <td>0.246365</td>
      <td>-0.415011</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.463328</td>
      <td>0.265354</td>
      <td>-0.513488</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.603284</td>
      <td>0.080577</td>
      <td>-0.470257</td>
    </tr>
  </tbody>
</table>
</div>




```python

```


```python
np.mean(tetra)
```




    C1    0.116703
    C2   -0.065265
    C3    0.023340
    dtype: float64




```python
#Do we need to normalize?
```


```python
# Let's see how well the data is distributed
```


```python
tetra.hist()
```




    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x000000000950B0F0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000000009470630>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x0000000009CE2E80>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000000004C18A58>]], dtype=object)




![png](output_13_1.png)



```python
# Not perfect, but still good enough not to need normalization
```


```python
#########
```


```python
# Let's build a test dataset to test our functions
```


```python
x1= np.array([1,1,1,8,8,8,12,12,12])
```


```python
x2 = np.array([1.5,2,3,8.8,9,7,12.5,14,13])
```


```python
x3= np.array([2,1.9,0.8,8.9,7,9,11,13,14])
```


```python
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1, x2, x3, c='r', marker='o')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
```


![png](output_20_0.png)



```python
# Now lets build the tools we need to put together the kmeans
```


```python

```


```python

```


```python
# First we need to generate centroids
```


```python
# The method I will use is to shuffle the data & pick k points at random to be our centroids 
```


```python
#Put together an array from our data
```


```python
testarray= np.column_stack((x1,x2,x3))
```


```python
testarray
```




    array([[  1. ,   1.5,   2. ],
           [  1. ,   2. ,   1.9],
           [  1. ,   3. ,   0.8],
           [  8. ,   8.8,   8.9],
           [  8. ,   9. ,   7. ],
           [  8. ,   7. ,   9. ],
           [ 12. ,  12.5,  11. ],
           [ 12. ,  14. ,  13. ],
           [ 12. ,  13. ,  14. ]])




```python
np.shape(testarray)
```




    (9, 3)




```python
# shuffle the data, pick 3 rows , and make them our centroids
```


```python
#Take the array of our data
#Shuffle it
#take k points as the centroids 


def initialize_centroids(dataarray,k):
    shuffled = np.copy(dataarray)
    np.random.shuffle(shuffled)
    centroids = shuffled[0:k,]
    return(centroids)
```


```python
initialize_centroids(testarray,3)
```




    array([[  1. ,   3. ,   0.8],
           [ 12. ,  14. ,  13. ],
           [  8. ,   8.8,   8.9]])




```python
#Now we call the function and bind its output (which is our centroids) to the variable centroids
#We will be calling this variable a lot so don't screw it up
centroids = initialize_centroids(testarray,3)
```


```python
centroids
```




    array([[ 8. ,  8.8,  8.9],
           [ 8. ,  7. ,  9. ],
           [ 1. ,  2. ,  1.9]])




```python
# First we will build the tools we need to give each point a label
#This constitutes:
    # A function to calculate the distance between a point and another point (in this case a centroid)
    # A function to determine what's the closest centroid to a point
    # A loop that calculates for each point how far it is from each centroid, and then assigns it a label to the closest one
```


```python

```


```python
# Distance between a point and another point (in this case a centroid)
```


```python
def dist_between_2_points(point1,point2):   
    return np.sqrt(np.sum((point1-point2)**2))



```


```python
dist_between_2_points(testarray[0],centroids[0])
```




    12.243365550370537




```python
# Find the closest centroid to a point and reference it by which element of the centroids array it is
```


```python
def closest_centroid_for_this_point(this_point,centroid_array):
    distances_for_this_point = []
    for i in range(len(centroid_array)):
        distances_for_this_point.append(dist_between_2_points(this_point,centroid_array[i]))
    return(distances_for_this_point.index(min(distances_for_this_point)))
```


```python
# Test the function by feeding it a point and our array of centroids
```


```python
closest_centroid_for_this_point(testarray[0],centroids)
```




    2




```python
# Now we need a function that calculates the nearest centroid for each data point,
# and then compliles a list of labels 
```


```python
# Returns a list containing all labels for all points
def assign_labels(array_of_points,centroid_array):
    labels = []
    for i in range(array_of_points.shape[0]):
        labels.append(closest_centroid_for_this_point(array_of_points[i],centroid_array=centroid_array))
    return(labels)
```


```python
assign_labels(testarray,centroids)
```




    [2, 2, 2, 0, 0, 1, 0, 0, 0]




```python

```


```python
# This command generates the list of labels based on current centroid positions
# It will be called when determining new centroid positions
```


```python
current_labels = assign_labels(testarray,centroids)
```


```python

```


```python
# now we need functions to get geometric mean of a set of points (will be used to calculate the new centroid of a cluster)
```


```python
def geom_centroid_of_points(set_of_points):
    return((set_of_points).mean(axis=0))
```


```python
#to test this: I choose 3 points to be our array and one centroid to be that centroid 
```


```python
testarray[0:3]
```




    array([[ 1. ,  1.5,  2. ],
           [ 1. ,  2. ,  1.9],
           [ 1. ,  3. ,  0.8]])




```python
geom_centroid_of_points(testarray[0:3])
```




    array([ 1.        ,  2.16666667,  1.56666667])




```python

```


```python
# Now we need a way to change the position of a centroid to the geometric mean of the points near it
```


```python
# get array containing the points near a centroid
```


```python
def findindexes_in_cluster(list_of_values, target_value):
    return [i for i, x in enumerate(list_of_values) if x==target_value ]
```


```python
# Test, which points in the dataset belong to cluster 0 ?
findindexes_in_cluster(current_labels,0)
```




    [3, 4, 6, 7, 8]




```python
# Now we need a function that when we give it a cluster, retrieves the points that belong to that cluster,
# then returns thier geometric mean
```


```python
#indexes_for_cluster_0 = findindexes_in_cluster(current_labels,0)
```


```python
#testarray[indexes_for_cluster_0,:]
```


```python
#testarray[[1,2,3],:]
```


```python
#geom_centroid_of_points(testarray[indexes_for_cluster_0,:])
```


```python
def new_position_for_centroid(index_of_centroid,dataset):
    indexes_for_nearby_points = findindexes_in_cluster(current_labels,index_of_centroid)
    nearby_points = dataset[indexes_for_nearby_points,:]
    mean_of_nearby_points = geom_centroid_of_points(nearby_points)
    return(mean_of_nearby_points)
```


```python
# Test by giving it a centroid 0 and making sure it takes that centroid index, finds it's reference in the labels list,
# then then gets the mean of all points that share that label reference and returns it 
new_position_for_centroid(0,testarray)
```




    array([ 10.4 ,  11.46,  10.78])




```python

```


```python
# now we need a function that generates an array of new centroid positions
```


```python
# loop to update centroid positions
```


```python
for i in range(centroids.shape[0]):
    new_centroids = np.empty_like(centroids)
    new_centroids[i] = new_position_for_centroid(i,testarray)
```


```python
new_centroids
```




    array([[ 10.4       ,  11.46      ,  10.78      ],
           [  0.        ,   0.        ,   0.        ],
           [  1.        ,   2.16666667,   1.56666667]])




```python
# now we have built all the tools we need
```


```python
# its time to put them all together into a kmeans function
```


```python
# I will repeat all the functions here so we have them all in one place
```


```python

```


```python
#Take the array of our data
#Shuffle it
#take the first k points as the centroids 


def initialize_centroids(dataarray,k):
    shuffled = np.copy(dataarray)
    np.random.shuffle(shuffled)
    centroids = shuffled[0:k,]
    return(centroids)
```


```python
def dist_between_2_points(point1,point2):   
    return np.sqrt(np.sum((point1-point2)**2))
```


```python
def closest_centroid_for_this_point(this_point,centroid_array):
    distances_for_this_point = []
    for i in range(len(centroid_array)):
        distances_for_this_point.append(dist_between_2_points(this_point,centroid_array[i]))
    return(distances_for_this_point.index(min(distances_for_this_point)))
```


```python
# Returns a list containing all labels for all points
def assign_labels(array_of_points,centroid_array):
    labels = []
    for i in range(array_of_points.shape[0]):
        labels.append(closest_centroid_for_this_point(array_of_points[i],centroid_array=centroid_array))
    return(labels)
```


```python
def geom_centroid_of_points(set_of_points):
    return((set_of_points).mean(axis=0))
```


```python
def findindexes_in_cluster(list_of_values, target_value):
    return [i for i, x in enumerate(list_of_values) if x==target_value ]
```


```python
def new_position_for_centroid(index_of_centroid,dataset,current_labels):
    indexes_for_nearby_points = findindexes_in_cluster(current_labels,index_of_centroid)
    nearby_points = dataset[indexes_for_nearby_points,:]
    mean_of_nearby_points = geom_centroid_of_points(nearby_points)
    return(mean_of_nearby_points)
```


```python
def kmeans(dataarray,k,iterations):
    #first initialize the centroids
    centroids = initialize_centroids(dataarray,k)
    
    #
    for i in range(0,iterations):
        #based on where the centroids currently are, assign a label for each data point to its nearest centroid 
        current_labels = assign_labels(dataarray,centroids)
        #Based on these newly assigned labels moves the centroids to thier new positions
        #generate an empty array to house the new centroids, then populate that array with the new centroid positions
        new_centroids = np.empty_like(centroids)
        for i in range(centroids.shape[0]):
            new_centroids[i] = new_position_for_centroid(i,dataarray, current_labels)
        # If the centroids haven't changed, stop 
        if np.array_equal(centroids,new_centroids):
            print("model converged")
            break
        #assign the new centroid positions to the centroids
        centroids = new_centroids  
    return(current_labels)
```


```python
# Now let's test the kmeans function using our testarray
```


```python
# It should return a vector of cluster assignments
```


```python
del(current_labels)
```


```python
# use our test array , use 3 centroids , run for 10 iterations
current_labels = kmeans(testarray,3,50)
```

    model converged
    


```python
# This should give us a vector of cluster assignments
```


```python
current_labels
```




    [0, 0, 0, 2, 2, 2, 1, 1, 1]




```python
# Success!!
```


```python
# Now let's plot our results to see if it grouped correctly
```


```python
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

X = testarray[:,0]
Y = testarray[:,1]
Z = testarray[:,2]



ax.scatter(X ,Y, Z,  c=current_labels, marker='o')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
```


![png](output_93_0.png)



```python
# Now Let's use our kmeans algo to cluster the tetra data
```


```python
tetra[:5]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>C1</th>
      <th>C2</th>
      <th>C3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.295428</td>
      <td>0.050829</td>
      <td>-0.385217</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.409178</td>
      <td>-0.035191</td>
      <td>-0.251980</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.096803</td>
      <td>0.246365</td>
      <td>-0.415011</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.463328</td>
      <td>0.265354</td>
      <td>-0.513488</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.603284</td>
      <td>0.080577</td>
      <td>-0.470257</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Let's convert the dataset into a numpy array so it doesn't screw us when we try to do matrix operations
```


```python
numpytetra = tetra.as_matrix()
```


```python
numpytetra
```




    array([[ 1.295428,  0.050829, -0.385217],
           [ 1.409178, -0.035191, -0.25198 ],
           [ 1.096803,  0.246365, -0.415011],
           ..., 
           [-0.726249, -0.103244,  0.6943  ],
           [ 0.808596, -0.49264 ,  1.64937 ],
           [ 0.749291, -0.44784 ,  0.863555]])




```python
#Clear out the labels variable
```


```python
del(current_labels)
```


```python
current_labels = kmeans(numpytetra,3,100)
```

    model converged
    


```python
current_labels
```




    [1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     0,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     0,
     1,
     1,
     1,
     1,
     1,
     1,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0]




```python
# Now let's plot the tetra data with the clusterings to see what happened
```


```python
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

X = numpytetra[:,0]
Y = numpytetra[:,1]
Z = numpytetra[:,2]



ax.scatter(X ,Y, Z,  c=current_labels, marker='o')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
```


![png](output_104_0.png)



```python
#Success!!
```
