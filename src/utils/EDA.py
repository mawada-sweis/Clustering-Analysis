# Libraries 
import pandas as pd

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns


def plt_missing_data(data, str):
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
    plt.title('Missing Values ' + str + ' Cleaning')
    plt.xlabel('Column Name')
    plt.ylabel('Percentage of Missing Values')
    plt.xticks(rotation=90)
    plt.show()