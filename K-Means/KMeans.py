import random
import numpy as np
import matplotlib.pyplot as plt

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def initialize_centroids(X, k):
    return X[random.sample(range(X.shape[0]), k)]

def assign_clusters(X, centroids):
    clusters = []
    for point in X:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        cluster_index = np.argmin(distances)
        clusters.append(cluster_index)
    return np.array(clusters)

def update_centroids(X, clusters, k):
    new_centroids = []
    for i in range(k):
        cluster_points = X[clusters == i]
        if len(cluster_points) > 0:
            new_centroids.append(np.mean(cluster_points, axis=0))
        else:
            new_centroids.append(X[random.randint(0, X.shape[0] - 1)])
    return np.array(new_centroids)

def kmeans(X, k, max_iterations=100, tolerance=1e-4):
    centroids = initialize_centroids(X, k)

    for iteration in range(max_iterations):
        clusters = assign_clusters(X, centroids)

        new_centroids = update_centroids(X, clusters, k)

        if np.all(np.abs(new_centroids - centroids) < tolerance):
            print(f"Converged after {iteration + 1} iterations.")
            break

        centroids = new_centroids

    return centroids, clusters

# def assig(X):
#  x=[]
#  y=[]
#  for i in range(X):
#   x.append(X[0])
def plot_clusters(X, centroids, clusters, k):
    colors = ['r', 'g', 'b', 'y', 'c', 'm']  # You can expand this list if k > 6
    for i in range(k):
        points = X[clusters == i]
        plt.scatter(points[:, 0], points[:, 1], s=100, c=colors[i], label=f"Cluster {i+1}")
    plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='black', marker='X', label='Centroids')
    plt.title(f'K-Means Clustering with {k} Clusters')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.show()

if __name__ == "__main__":
   
    X = np.array([
        [1.0, 2.1], [1.5, 1.8], [5.0, 8.0], [8.0, 8.0], 
        [1.2, 0.5], [9.0, 11.0], [8.0, 2.0], [10.0, 2.0], 
        [9.0, 3.0], [2.0, 2.0], [3.0, 3.0], [10.0, 9.0], 
        [6.0, 7.0], [8.0, 1.0], [3.5, 5.0], [6.5, 5.5],
        [7.2, 7.1], [3.9, 3.2], [2.5, 2.8], [7.5, 8.5],
        [4.2, 7.0], [8.9, 8.9], [6.3, 1.7], [9.3, 9.7],
        [1.7, 3.4], [3.4, 1.2], [5.7, 2.8], [9.8, 7.6],
        [6.2, 8.2], [3.6, 2.5], [7.8, 6.5], [8.1, 4.0],
        [2.7, 4.1], [4.5, 5.5], [1.0, 1.5], [7.7, 9.5],
        [4.9, 8.5], [3.2, 3.9], [5.9, 6.8], [6.0, 2.3],
        [9.6, 5.4], [1.5, 4.8], [2.9, 5.0], [4.8, 1.3],
        [5.2, 6.1], [6.7, 9.0], [7.0, 3.0], [9.1, 6.7]
    ])
    # q,w=assig(X)
    k = 3  

    
    centroids, clusters = kmeans(X, k)

    print("Final Centroids:", centroids)
    print("Cluster Assignments:", clusters) 

if __name__ == "__main__":
    k = 3  # Number of clusters
    centroids, clusters = kmeans(X, k)

    print("Final Centroids:", centroids)
    print("Cluster Assignments:", clusters)
    plot_clusters(X, centroids, clusters, k)
    
    

