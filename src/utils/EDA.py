# Libraries 
import pandas as pd
import numpy as np 

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


import numpy as np
import matplotlib.pyplot as plt

def create_histogram(data, bin=30, orientation='vertical', sort=True, xlabel='', ylabel='', title=''):
    """
    Creates a histogram of the given data.

    Args:
        data (list or numpy array): The data to create a histogram.
        bin (int): The num of bins
        orientation (str, optional): The orientation of the histogram, either 'vertical' (default) or 'horizontal'.
        sort (bool, optional): Whether to sort the values in data before creating the histogram. Defaults to True.
        xlabel (str, optional): The label for the x-axis.
        ylabel (str, optional): The label for the y-axis.
        title (str, optional): The title of the histogram.

    Returns:
        None
    """
    if sort:
        data = np.sort(data)

    plt.hist(data, bins=bin, orientation=orientation)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def create_stacked_histogram(df, column1, column2, bins=10, sort=True, title="", xlabel="", ylabel="Frequency", orientation="vertical"):
    """
    Creates a stacked histogram of the given columns in the DataFrame.

    Args:
        df (pandas DataFrame): The DataFrame to create the histogram for.
        column1 (str): The name of the first column to plot.
        column2 (str): The name of the second column to plot.
        bins (int, optional): The number of bins to use in the histogram.
        sort (bool, optional): Whether to sort the data before creating the histogram. Defaults to True.
        orientation (str, optional): The orientation of the histogram, either 'vertical' (default) or 'horizontal'.
        xlabel (str, optional): The label for the x-axis.
        ylabel (str, optional): The label for the y-axis.
        title (str, optional): The title of the histogram.

    Returns:
        None
    """
    if sort:
        df = df[[column1, column2]].apply(lambda x: np.sort(x), axis=0)
    
    # Create histogram of first column
    plt.hist(df[column1], bins=bins, alpha=0.5, label=column1, orientation=orientation)

    # Create histogram of second column
    plt.hist(df[column2], bins=bins, alpha=0.5, label=column2, orientation=orientation)

    # Add legend and labels
    plt.legend(loc='upper right')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()




def create_boxplot(df, col_name):
    """
    Create a boxplot for a given column of a pandas DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame to plot.
        col_name (str): The name of the column to plot.

    Returns:
        None
    """
    fig, ax = plt.subplots()
    ax.boxplot(df[col_name].dropna())

    plt.text(x = 1.2, y=df[col_name].min(), s='min')
    plt.text(x = 1.2, y=df[col_name].quantile(0.25), s='Q1')
    plt.text(x = 1.2, y=df[col_name].median(), s='median(Q2)')
    plt.text(x = 1.2, y=df[col_name].quantile(0.75), s='Q3')
    plt.text(x = 1.2, y=df[col_name].max(), s='max')

    #add the graph title and axes labels
    ax.set_title(f'Boxplot of {col_name}')
    ax.set_ylabel(col_name)
    plt.show()



def create_scatterplot(x, y, xlabel='', ylabel='', title=''):
    """
    Creates a scatter plot of the given x and y data.

    Args:
        x (list or numpy array): The x-axis data.
        y (list or numpy array): The y-axis data.
        xlabel (str, optional): The label for the x-axis.
        ylabel (str, optional): The label for the y-axis.
        title (str, optional): The title of the scatter plot.

    Returns:
        None
    """
    plt.scatter(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
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

