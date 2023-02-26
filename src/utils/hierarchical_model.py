from numpy import ndarray
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt


def plot_dendrogram_all_methods(data: pd.DataFrame, method_list=None, distance_metric='euclidean', optimal_ordering=False) -> None:
    """
    Calculates the linkage matrix and draws the dendrogram plot for each given linkage method.

    Args:
        data (pd.DataFrame): The dataset to draw the dendrogram plot of.
        method_list (list): The linkage methods to use. Defaults to ['single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward'].
        distance_metric (str): The distance metric to use when calculating the linkage matrix. Defaults to 'euclidean'.
        optimal_ordering (bool): Whether to use optimal leaf ordering when drawing the dendrogram plot. Defaults to False.
    """

    # Set default linkage methods if not provided
    if method_list is None:
        method_list = [
            'single',
            'complete',
            'average',
            'weighted',
            'centroid',
            'median',
            'ward',
        ]

    # Create a figure to plot the dendrograms
    fig, axs = plt.subplots(len(method_list), 1, figsize=(10, 30), sharex=True, gridspec_kw={'hspace': 0.5})

    # Loop over the methods and plot the corresponding dendrogram
    for index, method in enumerate(method_list):
        linkage_matrix = linkage(data, method=method, metric=distance_metric, optimal_ordering=optimal_ordering)
        dendrogram(linkage_matrix, ax=axs[index])
        axs[index].set_title(f'Dendrogram ({method} linkage)')
        axs[index].set_xlabel('Sample Index')
        axs[index].set_ylabel('Distance')

    plt.show()


def display_dendrogram(data: pd.DataFrame, method_type='ward', distance_metric='euclidean', optimal_ordering=False) -> ndarray:
    """
    Calculates the linkage matrix and draws the dendrogram plot.

    Args:
        data (pd.DataFrame): The dataset to draw the dendrogram plot of.
        method_list (str): The linkage method to use. Defaults to 'ward'.
        distance_metric (str): The distance metric to use when calculating the linkage matrix. Defaults to 'euclidean'.
        optimal_ordering (bool): Whether to use optimal leaf ordering when drawing the dendrogram plot. Defaults to False.

    Returns:
        scipy.cluster.hierarchy.dendrogram: The dendrogram object.
    """

    # Calculate the linkage matrix
    linkage_matrix = linkage(data, method=method_type, metric=distance_metric, optimal_ordering=optimal_ordering)

    # Plot the dendrogram
    dendrogram(linkage_matrix)

    # Set the plot title and axis labels
    plt.title(f'Dendrogram ({method_type} linkage)')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')

    # Show the plot and return the dendrogram object
    plt.show()
    return linkage_matrix


def plot_thresholded_dendrogram(linkage_matrix: ndarray, distance_threshold: int) -> None:
    """
    Plots the dendrogram with a horizontal line at the given threshold.

    Args:
        linkage_matrix (numpy.ndarray): The linkage matrix.
        threshold (float): The threshold distance to plot the horizontal line at.
    """

    plt.figure(figsize=(10, 7))
    plt.title("Dendrogram with Threshold")
    dendrogram(linkage_matrix)
    plt.axhline(y=distance_threshold, color='black', linestyle='--')
    plt.show()
