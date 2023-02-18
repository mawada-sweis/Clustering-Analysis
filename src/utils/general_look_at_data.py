import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def get_feature_types_stats(df):
    """
    Computes the data types of features in a DataFrame.

    Args:
        df (pandas.DataFrame): The input DataFrame.

    Returns:
        pandas.DataFrame: A DataFrame containing the following columns:
        - feature: the name of the feature
        - feature_type: the data type of the feature
    """
    # Compute the data types of features in the DataFrame
    return df.dtypes.rename('feature_type').reset_index().rename(columns={'index': 'feature'})


def get_feature_missing(data):
    """
    Computes missing value statistics for each feature in a DataFrame.

    Args:
        df (pandas.DataFrame): The input DataFrame.

    Returns:
        pandas.DataFrame: A DataFrame containing the missing data statistics columns:
    """
    # Compute missing value counts and percentages for each feature
    missing_stats = data.isna().sum().rename('missing_count').reset_index()
    missing_stats['missing_percentage'] = missing_stats['missing_count'] / len(data)

    # Compute counts and percentages for each value category
    counts = {}
    for val, label in [(998, 'dont_know'), (-9998, 'no_response'), (9999, 'technological_error'), (999, 'prefer_not_to_answer'), (995, 'not_required')]:
        counts[label] = data[data == val].notna().sum().rename(f'{label}_count').reset_index()
        missing_stats[f'{label}_percentage'] = counts[label][f'{label}_count'] / len(data)

    # Add feature type information
    feature_types_df = get_feature_types_stats(data)
    missing_stats = missing_stats.rename(columns={'index': 'feature'}).merge(feature_types_df,
                                                                             on='feature',
                                                                             how='left')
    # set 'feature' column as index
    missing_stats.set_index('feature', inplace=True)
    
    # Filter rows where at least one value in the columns (except 'feature_type') is greater than 0
    filter_not_zero_percentage = missing_stats.drop(columns='feature_type').gt(0).any(axis=1)
    
    # Sort the results by descending order of missing_percentage
    return missing_stats[filter_not_zero_percentage].sort_values('missing_percentage', ascending=False)


def display_direct_missing(df):
    """
    Filters a pandas DataFrame to display only the 
    columns that have missing values and their corresponding 
    percentage of missing values.

    Args:
        df (pandas DataFrame): The DataFrame to be filtered.

    Returns:
        pandas DataFrame or str: The filtered missing value 
        statistics DataFrame, or a string indicating that 
        there are no missing data of type NAN.    
    """
    # Filter rows with missing percentage greater than 0
    filtered_stats = df[df['missing_percentage'] > 0]
    
    # Get first two columns
    filtered_stats = filtered_stats.iloc[:, :2]
    
    # Check if there no missing data
    if filtered_stats.shape[0] == 0:
        filtered_stats = 'No Missing Data: NAN type'

    # Return the filtered missing value statistics
    return filtered_stats


def display_indirect_missing(df):
    # Filter rows with missing percentage greater than 0
    filtered_stats = df[df.drop(columns='feature_type').gt(0).any(axis=1)]

    # Return indirect missing columns
    return filtered_stats.drop(columns=['missing_count', 'missing_percentage'])


def plot_indirect_missing(missing_data):
    # Convert non-numeric values to NaN
    missing_data = missing_data.apply(pd.to_numeric, errors='coerce')

    # Filter features where the percentage is greater than zero
    filtered_data = missing_data[missing_data.gt(0.0).any(axis=1)].drop(columns='feature_type')

    # Create a bar plot of the missing data percentage for each feature
    num_features = len(filtered_data.columns)
    num_cols = 2
    num_rows = math.ceil(num_features / num_cols)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 5*num_rows))

    for i, column_name in enumerate(filtered_data.columns):
        try:
            filtered2_data = filtered_data[column_name][filtered_data[column_name].gt(0.0)].reset_index()
            row_idx = i // num_cols
            col_idx = i % num_cols
            ax = axs[row_idx][col_idx]
            sns.barplot(x=column_name, y='feature', data=filtered2_data, ax=ax)
            ax.set_title(f'Missing Data by {column_name}')
            ax.set_xlabel('Missing Data Percentage')
            ax.set_ylabel('Feature')
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=5)

        except ValueError:
            print(f"No valid data found for {column_name}")
    
    plt.tight_layout()
    plt.show()