from numpy import ndarray
import pandas as pd
import numpy as np

from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from sklearn.metrics import silhouette_samples, silhouette_score

# Define a list of colors to choose from
colors_code = ['#609EA2', '#AA5656', '#F2921D', "#4E6C50"]


def get_linkage(data: pd.DataFrame, method_type='ward', distance_metric='euclidean', optimal_ordering=False) -> ndarray:
    """Calculates hierarchical clustering linkage matrix for a given dataset.

    Args:
        data (pandas.DataFrame): The dataset to cluster.
        method_type (str, optional): The clustering algorithm to use. Default is 'ward'.
        distance_metric (str, optional): The distance metric to use. Default is 'euclidean'.
        optimal_ordering (bool, optional): Whether to compute the optimal leaf ordering for the dendrogram. Default is False.

    Returns:
        linkage_matrix (numpy.ndarray): The hierarchical clustering linkage matrix.
    """
    return linkage(data, method=method_type, metric=distance_metric, optimal_ordering=optimal_ordering)


def plot_dendrogram_all_methods(data: pd.DataFrame, method_list=None, distance_metric='euclidean', optimal_ordering=False) -> None:
    """Calculates the linkage matrix and display the dendrogram plot for each given linkage method.

    Args:
        data (pd.DataFrame): The dataset to draw the dendrogram plot of.
        method_list (list): The linkage methods to use. Defaults to ['single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward'].
        distance_metric (str): The distance metric to use when calculating the linkage matrix. Defaults to 'euclidean'.
        optimal_ordering (bool): Whether to use optimal leaf ordering when drawing the dendrogram plot. Defaults to False.
    """

    # Set default linkage methods if not provided
    if method_list is None:
        method_list = ['single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward']

    # Create a figure to plot the dendrograms
    fig, axs = plt.subplots(len(method_list), 1, figsize=(10, 30), sharex=True, gridspec_kw={'hspace': 0.5})

    # Loop over the methods and plot the corresponding dendrogram
    for index, method in enumerate(method_list):
        # Calculate the linkage matrix
        linkage_matrix = get_linkage(data, method, distance_metric, optimal_ordering)
        
        # Plot the dendrogram
        dendrogram(linkage_matrix, ax=axs[index])
        
        # Set the plot title and axis labels
        axs[index].set_title(f'Dendrogram ({method} linkage)')
        axs[index].set_xlabel('Sample Index')
        axs[index].set_ylabel('Distance')
    
    # Show the plot and return the dendrogram object
    plt.show()


def display_dendrogram(linkage_matrix: pd.DataFrame, method_type='ward') -> None:
    """Display the dendrogram plot given a linkage matrix and a linkage method. 
    The dendrogram plot shows the hierarchical clustering of the dataset.

    Args:
        linkage_matrix (pd.DataFrame): The linkage matrix of the dataset.
        method_type (str): The linkage method to use. Defaults to 'ward'.
    """
    # Plot the dendrogram
    dendrogram(linkage_matrix)

    # Set the plot title and axis labels
    plt.title(f'Dendrogram ({method_type} linkage)')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')

    # Show the plot
    plt.show()


def plot_thresholded_dendrogram(linkage_matrix: ndarray, distance_threshold: int) -> None:
    """Plots the dendrogram with a horizontal line at the given threshold.

    Args:
        linkage_matrix (numpy.ndarray): The linkage matrix of the dataset.
        threshold (float): The threshold distance to plot the horizontal line at.
    """
    # Plot the dendrogram
    plt.figure(figsize=(10, 7))
    dendrogram(linkage_matrix)
    
    # Set the plot title and axis labels
    plt.title("Dendrogram with Threshold")
    
    # Draw the horizontal line
    plt.axhline(y=distance_threshold, color='black', linestyle='--')
    
    # Show the plot
    plt.show()


def hierarchical_clustering(linkage_matrix: ndarray, k: int, criterion='maxclust') -> ndarray:
    """To cluster one-hot encoded data using fcluster from hierarchical clustering algorithms.

    Args:
        linkage_matrix (ndarray): The linkage matrix of the dataset.
        k (int): number of clusters.
        criterion (str, optional): The criterion to use in forming flat clusters. Defaults to 'maxclust'.

    Returns:
        ndarray: An array of length n. T[i] is the flat cluster number to which original observation i belongs.
    """
    return fcluster(linkage_matrix, k, criterion)


def _map_color_cluster(df: pd.DataFrame) -> pd.Series(str):
    """Assigns a color to each cluster in the given DataFrame based on their label.
    Args:
        df (pd.DataFrame): The DataFrame containing the cluster labels.

    Returns:
        pd.Series(str): A list of colors, where the i-th element corresponds to the color of the i-th row in the DataFrame.
    """
    # Get the unique cluster labels from the DataFrame
    cluster_labels = df['cluster'].unique()

    # Assign a color to each cluster label
    cluster_colors = {label: colors_code[i] for i, label in enumerate(cluster_labels)}

    # Create a new column in the DataFrame to store the cluster colors
    return df['cluster'].apply(lambda x: cluster_colors[x])


def get_transform_PCA(data: pd.DataFrame, dimensions: int) -> ndarray:
    """Performs principal component analysis (PCA) on a given pandas DataFrame data
        and returns the transformed data.

    Args:
        data (pd.DataFrame): The input data to be transformed using PCA.
        dimensions (int): The number of dimensions to keep after the transformation.
    Returns:
        ndarray: A numpy ndarray of shape (n_samples, dimensions) representing the transformed data.
    """
    # Perform PCA on the data
    pca = PCA(n_components=dimensions)
    return pca.fit_transform(data)


def plot_clustering_before_reduction(data: pd.DataFrame, dimensions: int) -> None:
    """Generates a scatter plot of the clustering of data before reducing its dimensionality using PCA.

    Args:
        data (pd.DataFrame): The input dataset to be plotted.
        dimensions (int): The number of dimensions to reduce the data to before plotting.
    Raises:
        ValueError: To check the requested dimensions are either 2 or 3.
    """
    # Check the requested dimensions
    if dimensions not in [2, 3]:
        raise ValueError("dimensions must be either 2 or 3")
    
    # Perform PCA on the data
    transformed = get_transform_PCA(data, dimensions)
    
    # Assigns a color to each cluster in the given DataFrame based on their label.
    colors = _map_color_cluster(data)
    
    # Set up plot
    fig = plt.figure()
    
    # Plot scatter by PCA transformed data
    if dimensions == 2:
        ax = fig.add_subplot(111)
        ax.scatter(transformed[:, 0], transformed[:, 1], c=colors)
    else:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(transformed[:, 0], transformed[:, 1], transformed[:, 2], c=colors)
        ax.set_zlabel('PC3')

    # Set axis labels and title
    ax.set_title(f'{dimensions}-Dimension Clustering before PCA')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    
    # Show the plot
    plt.show()


def plot_clustering_after_reduction(data: pd.DataFrame, transformed: ndarray, dimensions: int) -> None:
    """Generates a scatter plot of the clustering of data after reducing using the given transformed data by PCA.

    Args:
        data (pd.DataFrame): The input dataset to be plotted.
        transformed (ndarray): The transformed data using PCA. 
        dimensions (int): The number of dimensions to reduce the data to after plotting.
    Raises:
        ValueError: To check the requested dimensions are either 2 or 3.
    """
    # Check the requested dimensions
    if dimensions not in [2, 3]:
        raise ValueError("dimensions must be either 2 or 3")
    
    # Assigns a color to each cluster in the given DataFrame based on their label.
    colors = _map_color_cluster(data)
    
    # Set up plot
    fig = plt.figure()
    if dimensions == 2:
        ax = fig.add_subplot(111)
        ax.scatter(transformed[:, 0], transformed[:, 1], c=colors)
    
    else:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(transformed[:, 0], transformed[:, 1], transformed[:, 2], c=colors)
        ax.set_zlabel('PC3')

    # Set axis labels and title
    ax.set_title(f'{dimensions}-Dimension Clustering after PCA')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    
    # Show the plot
    plt.show()


def get_silhouette_vals_avg(data:pd.DataFrame):
    """Compute the silhouette values and average silhouette score for a clustering result.

    Args:
        data (pd.DataFrame): A DataFrame with the clustering result in a column named 'cluster' and the feature matrix in the other columns.

    Returns:
        A tuple containing the cluster labels, an array of silhouette values for each point, and the average silhouette score.
    """
    # Extract the feature matrix and cluster labels from the input data
    X = data.drop('cluster', axis=1)
    labels = data['cluster']

    # Compute the silhouette values for each point
    silhouette_vals = silhouette_samples(X, labels)

    # Compute the average silhouette score
    silhouette_avg = silhouette_score(X, labels)

    # Return the cluster labels, silhouette values, and average score as a tuple
    return labels, silhouette_vals, silhouette_avg


def plot_silhouette(data:pd.DataFrame) -> None:
    """Plot the Silhouette values for each cluster in a clustering result.

    Args:
        data (pd.DataFrame): A DataFrame with the clustering result in a column named 
            'cluster' and the feature matrix in the other columns.
    """
    # Get the cluster labels, Silhouette values, and average score from the input data
    labels, silhouette_vals, silhouette_avg = get_silhouette_vals_avg(data)

    # Create a new plot with a specified size
    fig, ax = plt.subplots(figsize=(10, 6))

    # Set the lower y-value for the first cluster
    y_lower = 10

    # Iterate over the unique cluster labels
    for i in np.unique(labels):
        # Select the Silhouette values for the current cluster label
        ith_cluster_silhouette_vals = silhouette_vals[labels == i]

        # Sort the Silhouette values for the current cluster label
        ith_cluster_silhouette_vals.sort()

        # Compute the size of the current cluster label
        size_cluster_i = ith_cluster_silhouette_vals.shape[0]

        # Set the upper y-value for the current cluster label
        y_upper = y_lower + size_cluster_i

        # Choose a color for the current cluster label
        color = plt.cm.nipy_spectral(float(i) / np.max(labels))

        # Fill the area between the lower and upper y-values with the Silhouette values for the current cluster label
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                         0, ith_cluster_silhouette_vals,
                         facecolor=color, edgecolor=color, alpha=0.7)

        # Label the current cluster label at the middle of the Silhouette values
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Set the lower y-value for the next cluster label
        y_lower = y_upper + 10

    # Set the x-label to Silhouette coefficient values
    ax.set_xlabel("Silhouette coefficient values")

    # Set the y-label to Cluster labels
    ax.set_ylabel("Cluster labels")

    # Add a vertical line at the average Silhouette score
    ax.axvline(x=silhouette_avg, color="black", linestyle="--")

    # Set the x-axis limits
    ax.set_xlim([-0.1, 1])

    # Highlight the cluster with the largest silhouette_avg
    max_index = np.argmax(silhouette_avg)
    ax.get_children()[max_index].set_facecolor('red')

    # Show the plot
    plt.show()


def plot_avg_silhouette(datasets) -> None:
    """Plots the average silhouette coefficient for a list of datasets and
        highlights the dataset with the maximum value.

    Args:
        datasets (list of pd.DataFrame): A list of datasets where each dataset 
            is a Pandas DataFrame with a 'cluster' column.
    """
    # Create a new figure with a single subplot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Initialize the maximum average to zero
    max_avg = 0

    # Iterate through each dataset and calculate its silhouette coefficient average
    for i, data in enumerate(datasets):
        # Extract the feature matrix X and the cluster labels from the dataset
        X = data.drop('cluster', axis=1)
        labels = data['cluster']

        # Calculate the silhouette coefficient average for this dataset
        silhouette_avg = silhouette_score(X, labels)

        # Add a scatter point for this dataset's silhouette coefficient average
        ax.scatter(i, silhouette_avg, s=100, c='blue')

        # If this dataset has the highest silhouette coefficient average so far, save its index and value
        if silhouette_avg > max_avg:
            max_avg = silhouette_avg
            max_idx = i

    # Add a scatter point to highlight the dataset with the highest silhouette coefficient average
    ax.scatter(max_idx, max_avg, s=200, c='red', marker='*')

    # Set the x-ticks and labels to indicate the dataset number
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels([f"Dataset {i+1}" for i in range(len(datasets))])

    # Set the x- and y-axis labels
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Silhouette coefficient average")

    # Display the plot
    plt.show()
