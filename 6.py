import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
from scipy.spatial.distance import cdist

data = pd.read_csv('USA_Housing.csv')
# nulls = pd.DataFrame(data.isnull().sum().sort_values(ascending=False))
# nulls.columns = ['Null Count']
# nulls.index.name  = 'Feature'
# print(nulls)

print(data.columns)

# Index(['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
#        'Avg. Area Number of Bedrooms', 'Area Population', 'Price', 'Address'],
#       dtype='object')

house = data.select_dtypes(include=[np.number]).interpolate().dropna()
print(sum(house.isnull().sum()  != 0))
nulls = pd.DataFrame(house.isnull().sum().sort_values(ascending=False))
nulls.columns = ['Null Count']
nulls.index.name  = 'Feature'
print(nulls)

print(' correlation of all columns are :\n' + str(data[data.columns[:]].corr()['Price'].sort_values(ascending=False)))

sns.FacetGrid(data,size=7)\
.map(plt.scatter,'Avg. Area Number of Rooms','Price')\
.add_legend()
plt.show()

x = house.iloc[:,[0,1,2,3,4,5]]
y = house.iloc[:,-1]
print(x.shape, y.shape)


from sklearn import preprocessing

scaler = preprocessing.StandardScaler()
from sklearn import metrics
scaler.fit(x)
X_scaled_array = scaler.transform(x)
X_scaled = pd.DataFrame(X_scaled_array, columns = x.columns)


from sklearn.cluster import KMeans
Sum_of_squared_distances = []
K = range(1,11)
for k in K:
    km = KMeans(n_clusters=k,init='k-means++',max_iter=300,n_init=10,random_state=0)
    km = km.fit(X_scaled)
    Sum_of_squared_distances.append(km.inertia_)


plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('The Elbow Method showing the optimal k')
plt.show()


nclusters = 3 # this is the k in kmeans
seed = 0
K = range(2,10)
for k in K:
    km = KMeans(n_clusters=k, random_state=seed)
    km.fit(X_scaled)
# predict the cluster for each data point
    y_cluster_kmeans = km.predict(X_scaled)
    plt.scatter(X_scaled_array[:, 2], X_scaled_array[:, 5], c=y_cluster_kmeans, s=50)
    centers = km.cluster_centers_
    plt.scatter(centers[:, 2], centers[:, 5], c='black', s=200, alpha=0.5)
    plt.show()
    score = metrics.silhouette_score(X_scaled, y_cluster_kmeans)
    print('silhouette score for clusters ' +str(k)+' is : ' + str(score))


# pca = PCA(n_components=2)
# pca.fit(X_scaled)
# X_pca = pca.transform(X_scaled)
# print("original shape:   ", X_scaled.shape)
# print("transformed shape:", X_pca.shape)
# X_new = pca.inverse_transform(X_pca)
# # plt.scatter(X_scaled[:, 0], X_scaled[:, 1], alpha=0.2)
# # plt.scatter(X_new[:, 0], X_new[:, 1], alpha=0.8)
# # plt.axis('equal');
# M = range(2,10)
# for m in M:
#     km = KMeans(n_clusters=m, random_state=seed)
#     km.fit(X_pca)
# # predict the cluster for each data point
#     y_cluster_kmeans_pca = km.predict(X_pca)
#
#     score1 = metrics.silhouette_score(X_pca, y_cluster_kmeans_pca)
#     print('silhouette score for clusters after PCA is ' +str(m)+' is : ' + str(score1))
#
