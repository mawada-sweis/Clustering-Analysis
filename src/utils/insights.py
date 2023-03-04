import matplotlib.pyplot as plt
import pandas as pd




def clusters_variances(clusters:list, columns:list)-> pd.DataFrame():
    """
    Calculates the variances of specified columns for each cluster in a list of dataframes, and returns the results as a pandas dataframe.

    Parameters:
    clusters (list): A list of pandas dataframes, where each dataframe represents a cluster of data points.
    columns (list): A list of strings representing the names of the columns for which to calculate the variances.

    Returns:
    A pandas dataframe with the variances of the specified columns for each cluster.

    """
    # empty data frame
    variances = pd.DataFrame()

    # for cluster label
    count = 0

    for cluster in clusters:
        if count == 0:
            variances['all_data'] = cluster[columns].var().sort_values(ascending=True)
            
        else: 
            variances[f'cluster_{count}'] = cluster[columns].var().sort_values(ascending=True)

        count += 1
    
    return variances



def calculate_percentage(cluster:pd.DataFrame, columns:list) -> pd.DataFrame():
 
    """
    Calculate the percentage of each unique value in a specified column of a Pandas DataFrame.

    Parameters:
        data (pd.DataFrame): Input DataFrame
        columns (list): List of column names for which the percentages are computed

    Returns:
        pd.DataFrame: DataFrame with percentage values for each unique value in each specified column
    """
    percentage_data = pd.DataFrame()
    
    cluster = cluster[columns]

    # Compute the percentage of occurrence for each unique value in the concatenated data
    percentage_data = cluster.apply(lambda col: col.value_counts(normalize=True) * 100)

    return percentage_data



def plot_hist(clusters:list, column:str):
    """
    Plots a histogram for a specified column of each cluster in a list of dataframes.

    Parameters:
    clusters (list): A list of pandas dataframes, where each dataframe represents a cluster of data points.
    column (str): The name of the column to plot the histogram for.

    Returns:
    None

    """
    count = 0
    for cluster in clusters:
        if count==0:
            plt.figure(figsize=(2,2))
            plt.hist(cluster[column], bins=20)
            plt.title(f'all_data_{column}')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.show()  

        else :
            plt.figure(figsize=(2,2))
            plt.hist(cluster[column])
            plt.title(f'cluster{count}_{column}')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.show()

            
        count +=1
    