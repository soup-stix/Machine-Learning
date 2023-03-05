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
    
def modelAccuracy(Y,y):
    return (np.sum(y == Y) / len(Y)) * 100