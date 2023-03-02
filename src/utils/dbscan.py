# libraries
import pandas as pd
import numpy as np
import sys

# modelling - dbscan
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_samples, silhouette_score

# dimensionality reduction 
from sklearn.decomposition import PCA

# visulaization
import matplotlib.pyplot as plt

def create_pca(data, n_components):
    """
    Apply Principal Component Analysis (PCA) to reduce the dimensionality of the input data.

    Args:
        data (array-like): The input data to be transformed.
        n_components (int): The number of principal components to keep.

    Returns:
        X_reduced (array-like): The reduced dataset obtained by applying PCA to the input data.
    """
    # Create a PCA object with the specified number of components
    pca = PCA(n_components=n_components)

    # Fit the PCA model to the input data and transform it
    X_reduced = pca.fit_transform(data)

    # Print the explained variance ratio for each component
    print("Explained variance ratio for each component:", pca.explained_variance_ratio_)

    # Return the reduced dataset
    return X_reduced

def get_kdistances(data, k):
    """
    Compute the k-distance for each data point in a dataset and plot the distances graph.

    Parameters:
        data (array-like): The dataset to compute distances for.
        k (int): The number of nearest neighbors to use.

    Prints k-distance graph

    Returns:
        None
    """
    # Create a nearest neighbors object and fit it to the dataset
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(data)

    # Compute the distances to the kth nearest neighbor for each data point
    distances, _ = neigh.kneighbors(data)
    distances = np.sort(distances[:, k-1], axis=0)

    # Plot the distances
    plt.plot(distances)
    plt.xlabel('Points')
    plt.ylabel(f"Distance to {k}th nearest neighbor")
    plt.show()


def perform_dbscan(X_reduced, eps, min_samples):
    """
    Perform DBSCAN clustering on a reduced dataset.

    Parameters:
        X_reduced (array-like): A 2D array of the reduced dataset.
        eps (float): The maximum distance between two samples for them to be considered as in the same neighborhood.
        min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.

    Returns:
        labels (array-like): An array of cluster labels.
    """
    # Create a DBSCAN object
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)

    # Fit the DBSCAN model to the reduced dataset
    dbscan.fit(X_reduced)

    # Extract the cluster labels
    labels = dbscan.labels_

    # Return the cluster labels
    return labels

def range_hyperparameters(dataset, eps_range, minPts_range):
    """
    Performs DBSCAN on a dataset with a range of eps and minPts values.
    
    Parameters:
        dataset (numpy.ndarray): The dataset to cluster.
        eps_range (list): A list of eps values to test.
        minPts_range (list): A list of minPts values to test.
        
    Returns:
        pandas.DataFrame: A dataframe containing the results of each DBSCAN run.
    """
    
    # Create empty dataframe to store results
    results_df = pd.DataFrame(columns=["eps", "minPts", "silhouette_score", "n_clusters"])

    # Loop through parameter ranges and perform DBSCAN on each combination
    for eps in eps_range:
        for minPts in minPts_range:

            # Initialize DBSCAN with current parameter values
            dbscan = DBSCAN(eps=eps, min_samples=minPts)

            # Fit DBSCAN to data
            labels = dbscan.fit_predict(dataset)

            # Calculate silhouette score and number of clusters
            if len(set(labels)) > 1:  # Skip if only one cluster is found
                score = silhouette_score(dataset, labels)
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            else:
                score = None
                n_clusters = 0

            # Drop row if number of clusters is 0 or 1
            if n_clusters < 2:
                continue

            # Add results to dataframe
            results = {"eps": eps, "minPts": minPts, "silhouette_score": score, "n_clusters": n_clusters}
            results_df = pd.concat([results_df, pd.DataFrame(results, index=[0])], ignore_index=True)

    return results_df