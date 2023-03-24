import numpy as np
from collections import Counter

class KNearestNeighbours:
    def __init__(self, k=3):
        self.k=k

    def euclidean_distance(self,x1,x2):
        distance = np.sqrt(np.sum((x1-x2)**2))
        return distance

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self,X):
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self,x):
        distances = [self.euclidean_distance(x,x_train) for x_train in self.X_train]

        k_indices = np.argsort(distances)[:self.k]
        k_nearest_label = [self.y_train[i] for i in k_indices]

        most_commom = Counter(k_nearest_label).most_common()
        return most_commom[0][0]
    

class LogisticRegression:
    def __init__(self, l=0.001, epochs=1000):
        self.l = l
        self.epochs = epochs
        self.weights = None
        self.bias = None
    
    def fit(self,X,y):
        n_samples, n_features = X.shape 
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.epochs):
            linear_predictions = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(linear_predictions)

            d_w = (1/n_samples) * np.dot(X.T, (predictions - y))
            d_b = (1/n_samples) * np.sum((predictions - y))

            self.weights = self.weights - self.l*d_w
            self.bias = self.bias - self.l*d_b

    def predict(self, X):
        linear_predictions = np.dot(X, self.weights) + self.bias
        y_predictions = self.sigmoid(linear_predictions)
        predictions = [0 if y<=0.5 else 1 for y in y_predictions]
        return predictions

    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
    
class LinearRegression:
    def __init__(self,l=0.001,epochs=5000):
        self.l = l
        self.epochs = epochs
        self.weight = None
        self.bias = None

    def fit(self,X,y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.epochs):
            y_prediction = np.dot(X, self.weights) + self.bias
            d_m = (-2/n_samples)*np.dot(X.T, (y-y_prediction))
            d_c = (-2/n_samples)*np.sum((y-y_prediction))
            self.weights = self.weights - self.l*d_m
            self.bias = self.bias - self.l*d_c

    def predict(self, X):
        linear_predictions = np.dot(X, self.weights) + self.bias
        return linear_predictions
    
    def mse(self,Y,y):
        return np.mean((Y-y)**2)
    
class SVM:
    def __init__(self,l=0.001,lam=0.01,epochs=1000):
        self.l = l
        self.lam = lam
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self,X,y):
        n_samples, n_features = X.shape
        y = np.where(y<= 0,-1,1)

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.epochs):
            for i,x in enumerate(X):
                condition = y[i]*(np.dot(x,self.weights)-self.bias)
                if condition:
                    self.weights -= self.l * (2*self.lam*self.weights)
                else:
                    self.weights -= self.l * (2*self.lam*self.weights - np.dot(x,y[i]))
                    self.bias -= self.l* y[i]

    def predict(self,X):
        predictions = np.dot(X,self.weights) - self.bias
        return np.sign(predictions)
    

class KMeans:
    def __init__(self,k = 5, epochs = 100):
        self.k = k
        self.epochs = epochs

        #indices of each clustures
        self.clusters = [[] for _ in range(self.k)]

        #mean of each cluster
        self.centroids = []

    def predict(self,X):
        self.X = X
        self.n_samples, self.n_features = X.shape

        random_samples = np.random.choice(self.n_samples, self.k, replace=False)
        self.centroids = [self.X[ind] for ind in random_samples]

        for _ in range(self.epochs):
            self.clusters = self.create_clusters(self.centroids)

            centroids_old = self.centroids
            self.centroids = self.new_centroids(self.clusters)

            if self.is_converged(centroids_old, self.centroids):
                break

        return self.get_cluster_lables(self.clusters)

    def create_clusters(self, centroids):
        clusters = [[] for _ in range(self.k)]
        for idx,sample in enumerate(self.X):
            centroid_idx = self.closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters
    
    def closest_centroid(self,sample,centroids):
        distances = [euclidean_distance(sample, point) for point in centroids]
        return np.argmin(distances)

    def new_centroids(self, clusters):
        centroids = np.zeros((self.k, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def is_converged(self, old, new):
        distances = [euclidean_distance(old[i],new[i]) for i in range(self.k)]
        return sum(distances) == 0

    def get_cluster_lables(self,clusters):
        lables = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                lables[sample_idx] = cluster_idx

        return lables
    
    def euclidean_distance(self,x1, x2):
        return np.sqrt(np.sum((x1-x2)**2))
    
def modelAccuracy(Y,y):
    return (np.sum(y == Y) / len(Y)) * 100