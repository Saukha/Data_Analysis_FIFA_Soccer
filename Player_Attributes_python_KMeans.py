import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans

# read data
Z = pd.read_csv("Player_Attributes.csv", header = 0)
# print('dataframe shape =', Z.shape)
# print(Z.head())
Z.dropna(inplace = True)
Z.drop(axis = 1, columns = ['id', 'player_fifa_api_id', 'player_api_id', 'date', 'preferred_foot', 'attacking_work_rate', 'defensive_work_rate'], inplace = True)
print('dataframe shape =', Z.shape)
# print(Z.head())
coln = Z.columns
# print(coln[0])
# print(type(coln[0]))

# draw smaller random a sample for analysis and graphing
Y = Z.sample(500)
print('sample dataframe shape =', Y.shape)

# create elbow graph
wcss = []  # initialize
for i in range(1, 8):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 15, n_init = 10, random_state = 0)
    kmeans.fit(Y)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 8), wcss, marker = 'o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()


# run k-means++ with k = 2
kmeans = KMeans(n_clusters = 2, init = 'k-means++', max_iter = 15, n_init = 10, random_state = 0)
kmeans.fit(Y)

# create graphs colored by cluster: Each attribute against Overall Rating 
for i in range(2,35):
    plt.scatter(Y.iloc[:,i], Y.iloc[:,0], c = kmeans.labels_, alpha = .5)
    plt.scatter(kmeans.cluster_centers_[:,i], kmeans.cluster_centers_[:,0], s=100, c = 'red', alpha = .7)
    plt.xlabel(coln[i])
    plt.ylabel(coln[0])
    title = "Player Attribute: " + coln[i] + " ~ Overall Rating"
    plt.title(title)
    plt.show()

