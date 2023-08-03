import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.optimize import dual_annealing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


def cda(X, y, n_clusters):
    # Perform clustering
    clusters = []
    for i in np.unique(y):
        kmeans = KMeans(n_clusters=n_clusters, n_init=10)
        kmeans.fit(X[y == i])
        clusters.append(kmeans.cluster_centers_)

    # Flatten the list of clusters
    clusters = np.concatenate(clusters)

    # Define the objective function, negative for numerical optimization
    def objective(u):
        u = u[:, np.newaxis]  # Reshape u to a column vector

        # Compute the mean of each cluster
        means = np.mean(clusters @ u, axis=0)

        # Compute the scatter within each cluster
        C = np.sum((clusters @ u - means) ** 2)

        # Compute the mean difference between clusters of different classes
        R = np.sum((means - np.mean(means)) ** 2)

        return -R / C

    # Numerical optimization method to maximise R/C, define the boundaries of the search space
    bounds = [(0, 100) for _ in range(X.shape[1])]

    # Optimisation using simulated annealing
    result = dual_annealing(objective, bounds)

    # Optimal projection vectors
    u_opti = result.x
    return u_opti


# Load the dataset
data = pd.read_csv('C:/Users/tkl68/Desktop/Iris/Iris.csv')

# Convert features into numpy arrays
X = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values

# Convert species labels into numpy arrays
le = LabelEncoder()
y = le.fit_transform(data['Species'])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Call the cda function
u_test = cda(X_train, y_train, n_clusters=3)

# Project data to lower dimensions using the optimal projection vector
X_train_projected = X_train @ u_test
X_test_projected = X_test @ u_test

# Reshape the projected data to 2D arrays
X_train_projected = X_train_projected.reshape(-1, 1)
X_test_projected = X_test_projected.reshape(-1, 1)

# Train a logistic regression classifier on the projected data
clf = LogisticRegression()
clf.fit(X_train_projected, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test_projected)

# Evaluate the classifier's performance
accuracy = accuracy_score(y_test, y_pred)
print(u_test)
print("Accuracy:", accuracy)
