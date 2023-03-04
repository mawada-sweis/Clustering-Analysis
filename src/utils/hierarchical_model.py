# Type hint
from numpy import ndarray

# Necessery libraries 
import pandas as pd
import numpy as np

# Hierarchical clustering
from scipy.cluster.hierarchy import dendrogram, linkage, fclusterdata

# visulaization
import matplotlib.pyplot as plt

# Dimensionality reduction
from sklearn.decomposition import PCA


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


def hierarchical_clustering(data: pd.DataFrame, k=3) -> ndarray:
    """To cluster one-hot encoded data using fcluster from hierarchical clustering algorithms.

    Args:
        linkage_matrix (ndarray): The linkage matrix of the dataset.
        k (int): number of clusters.
        criterion (str, optional): The criterion to use in forming flat clusters. Defaults to 'maxclust'.

    Returns:
        ndarray: An array of length n. T[i] is the flat cluster number to which original observation i belongs.
    """
    return fclusterdata(data, t=k, criterion='maxclust', method='ward')


def _map_color_cluster(labels: ndarray) -> pd.Series(str):
    """Assigns a color to each cluster in the given DataFrame based on their label.
    Args:
        labels(ndarray): Contains the label of clustering for each sample.
    Returns:
        pd.Series(str): A list of colors, where the i-th element corresponds to the color of the i-th row in the DataFrame.
    """
    # Get the unique cluster labels from the DataFrame
    cluster_labels = np.unique(labels)

    # Assign a color to each cluster label
    cluster_colors = {label: colors_code[i] for i, label in enumerate(cluster_labels)}
    
    # Create a new column in the DataFrame to store the cluster colors
    return pd.Series(labels).apply(lambda x: cluster_colors[x])


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


def plot_clustering(data: pd.DataFrame, labels:ndarray, dimensions: int, is_before: bool) -> None:
    """Generates a scatter plot of the clustering of data before reducing its dimensionality using PCA.

    Args:
        data (pd.DataFrame): The input dataset to be plotted.
        labels (ndarray):
        dimensions (int): The number of dimensions to reduce the data to before plotting.
        is_before (bool): 
    Raises:
        ValueError: To check the requested dimensions are either 2 or 3.
    """
    # Check the requested dimensions
    if dimensions not in [2, 3]:
        raise ValueError("dimensions must be either 2 or 3")
    
    if is_before:
        # Perform PCA on the data
        data = get_transform_PCA(data, dimensions)

    # Assigns a color to each cluster in the given DataFrame based on their label.
    colors = _map_color_cluster(labels)
    
    # Set up plot
    fig = plt.figure()
    
    # Plot scatter by PCA transformed data
    if dimensions == 2:
        ax = fig.add_subplot(111)
        ax.scatter(data[:, 0], data[:, 1], c=colors)
    else:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=colors)
        ax.set_zlabel('PC3')

    # Set axis labels and title
    ax.set_title(f'{dimensions}-Dimension Clustering before PCA')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    
    # Show the plot
    plt.show()
