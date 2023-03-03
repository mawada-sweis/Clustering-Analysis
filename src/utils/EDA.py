# Libraries 
import pandas as pd
import numpy as np 
from sklearn.decomposition import PCA

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns


def plt_missing_data(data, title):
    '''
    Plots a bar chart showing the percentage of missing values for each column.
    
    Args:
        data (pandas.DataFrame): The DataFrame to plot.
        str (str): A string to include in the chart title indicating the intended cleaning action.

    Returns:
        None. Displays a bar chart using matplotlib.
    '''
    # Calculate the percentage of missing values for each column
    missing_percentages = data.isnull().mean() * 100

    # Filter the columns that have missing values
    missing_columns = missing_percentages[missing_percentages > 0]

    # Sort the columns by percentage of missing values
    missing_columns = missing_columns.sort_values(ascending=False)

    # Plot the percentage of missing values for each column
    plt.bar(missing_columns.index, missing_columns.values)
    plt.title(title)
    plt.xlabel('Column Name')
    plt.ylabel('Percentage of Missing Values')
    plt.xticks(rotation=90)
    plt.show()


def create_histogram(data, bin=30, orientation='vertical', color='green', xlabel='', ylabel='', title=''):
    """
    Creates a histogram of the given data.

    Args:
        data (list or numpy array): The data to create a histogram.
        bin (int): The num of bins
        orientation (str, optional): The orientation of the histogram, either 'vertical' (default) or 'horizontal'.
        color (str, optional): The color of the histogram bars.
        xlabel (str, optional): The label for the x-axis.
        ylabel (str, optional): The label for the y-axis.
        title (str, optional): The title of the histogram.

    Returns:
        None
    """
    plt.hist(data, bins=bin, orientation=orientation, color=color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def create_boxplot(data, labels, color='green', xlabel='', ylabel='', title=''):
    """
    Creates a box plot of the given data.

    Args:
        data (list or numpy array): The data to create a box plot of.
        labels (list): The labels for the x-axis.
        color (str, optional): The color of the boxes and whiskers.
        xlabel (str, optional): The label for the x-axis.
        ylabel (str, optional): The label for the y-axis.
        title (str, optional): The title of the box plot.

    Returns:
        None
    """
    plt.boxplot(data, labels=labels, patch_artist=True, boxprops=dict(facecolor=color))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def create_scatterplot(x, y, xlabel='', ylabel='', title='', colors=None):
    """
    Creates a scatter plot of the given x and y data.

    Args:
        x (list or numpy array): The x-axis data.
        y (list or numpy array): The y-axis data.
        xlabel (str, optional): The label for the x-axis.
        ylabel (str, optional): The label for the y-axis.
        title (str, optional): The title of the scatter plot.
        colors (list or numpy array, optional): The color of each data point.

    Returns:
        None
    """
    if colors is None:
        plt.scatter(x, y)
    else:
        plt.scatter(x, y, c=colors, cmap='rainbow')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

def create_3d_scatter_plot(X, labels=None, cmap='viridis', fig_size=(10,10), title=None, xlabel=None, ylabel=None, zlabel=None):
    """
    Create a scatter plot of a dataset with more than 3 dimensions.

    Parameters:
        X (array-like): The dataset to plot.
        labels (array-like, optional): A list of labels for each data point.
        cmap (str, optional): The colormap to use for coloring the data points.
        fig_size (tuple, optional): The size of the figure (width, height) in inches.
        title (str, optional): The title of the plot.
        xlabel (str, optional): The label of the x-axis.
        ylabel (str, optional): The label of the y-axis.
        zlabel (str, optional): The label of the z-axis.
    """
    # Create a figure and axis object with 3D projection
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, projection='3d')

    # Plot the reduced dataset in a 3D scatter plot
    ax.scatter3D(X[:, 0], X[:, 1], X[:, 2], c=labels, cmap=cmap)

    # Add labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)

    # Show the plot
    plt.show()


def create_correlation_heatmap(df, title=''):
    """
    Creates a correlation heatmap of the given DataFrame.

    Args:
        df (pandas DataFrame): The DataFrame to create the correlation heatmap for.
        title (str, optional): The title of the heatmap.

    Returns:
        None
    """
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, square=True, annot=True, fmt='.2f',
                cbar_kws={"shrink": .8}, linewidths=.5, center=0, vmax=1, vmin=-1,
                annot_kws={"fontsize":8})
    plt.title(title)
    plt.show()


def create_PCA(data):
    """
    Creates a PCA plot of the given DataFrame.

    Args:
        df (pandas DataFrame): The DataFrame to create the correlation heatmap for.

    Returns:
        None
    """
    pca = PCA(n_components=7)
    pca.fit(data)
    variance = pca.explained_variance_ratio_ 
    var=np.cumsum(np.round(variance, 3)*100)
    plt.figure(figsize=(12,6))
    plt.ylabel('% Variance Explained')
    plt.xlabel('# of Features')
    plt.title('PCA Analysis')
    plt.ylim(0,100.5)
    plt.plot(var)
    plt.show()