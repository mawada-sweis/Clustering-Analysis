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


def display_missing_percentage(df):
    # Filter rows with missing percentage greater than 0
    filtered_stats = df[df['missing_percentage'] > 0]

    # Return first two columns
    return filtered_stats.iloc[:, :2]
