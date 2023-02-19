import pandas as pd
from IPython.display import display


def delete_columns(data, columns_name) -> None:
    """Delete all columns mensioned in the given columns name list.

    Args:
        data (pd.DataFrame): Current dataframe.
        columns_name (list): List of columns name that will be delete.
    """
    # Print the number of columns of the current dataset
    print("Current number of columns: ", data.shape[1])
    
    # Drop columns from the dataset
    data.drop(columns_name, axis='columns', inplace=True)

    # Print the number of columns of the updated dataset
    print("Updated number of columns: ", data.shape[1])
    

def filter_direct_missing_columns(df, lower_threshold=None, upper_threshold=None) -> list:
    """
    Filters a pandas DataFrame to return a list of the columns that have missing
    values either above a given upper threshold or within a given range between a 
    lower and upper threshold.

    Args:
        df (pandas DataFrame): The DataFrame to be filtered.
        lower_threshold (float): The minimum percentage of missing values allowed for columns to be included in the output. Default is None.
        upper_threshold (float): The maximum percentage of missing values allowed for columns to be included in the output. Default is None.

    Returns:
        list of str: The names of the columns that have missing values either above the upper threshold or within the specified percentage 
        range between the lower and upper thresholds.
    """
    # Filter the DataFrame based on the user-provided thresholds
    if lower_threshold is None:
        filtered_df = df[df['missing_percentage'] > upper_threshold]
    elif upper_threshold is None:
        filtered_df = df[df['missing_percentage'] < lower_threshold]
    else:
        filtered_df = df[
            (df['missing_percentage'] > lower_threshold) & 
            (df['missing_percentage'] < upper_threshold)
        ]

    # Return a list of the column names in the filtered DataFrame
    return list(filtered_df.index)


def fill_by_mode(data, columns_name) -> str:
    """
    Fill missing values in a pandas DataFrame column with the mode value.
    
    Args:
    - data: A pandas DataFrame with the column to fill
    - columns_name: The name of the column to fill
    
    Returns:
    - A string indicating the success of the operation.
    """
    
    # Fill missing values in the specified column using the mode
    mode_value = data[columns_name].mode().iloc[0,0]   # Compute the mode value
    data[columns_name] = data[columns_name].fillna(mode_value)   # Fill missing values with mode value
    
    # Return a success message
    return "Provided column successfully filled by mode value."


def filter_indirect_missing_columns(df, columns, lower_threshold=None, upper_threshold=None) -> list:
    """
    Filter a pandas DataFrame based on the percentage of indirect missing values for a list of columns.

    Args:
    - df: A pandas DataFrame
    - columns: A list of column names to filter
    - lower_threshold: A float indicating the lower threshold for the percentage of indirect missing values
    - upper_threshold: A float indicating the upper threshold for the percentage of indirect missing values

    Returns:
    - A list of column names in the filtered DataFrame
    """
    
    # Convert selected columns to numeric data type
    df[columns] = df[columns].apply(pd.to_numeric, errors='coerce')

    # Filter the DataFrame based on the user-provided thresholds for each column
    if lower_threshold is None:
        filtered_df = df.loc[df[columns].max(axis=1) > upper_threshold]

    elif upper_threshold is None:
        filtered_df = df.loc[df[columns].min(axis=1) < lower_threshold]
    
    else:
        filtered_df = df.loc[
            (df[columns].max(axis=1) > lower_threshold) & 
            (df[columns].min(axis=1) < upper_threshold)
        ]

    # Return a list of the column names in the filtered DataFrame
    return list(filtered_df.index)


def _is_one_hot_encoded(df, col_name):
    """
    Returns True if the given column is one-hot encoded in the DataFrame, and False otherwise.

    Args:
    - df: A pandas DataFrame
    - col_name: A string representing the name of the column

    Returns:
    - True if the column is one-hot encoded, and False otherwise
    """
    unique_vals = df[col_name].unique()

    # Check if the column has only two unique values
    if len(unique_vals) == 3:
        return True

    # Check if the column name contains an underscore followed by a unique value
    col_name_parts = col_name.split('_')
    if len(col_name_parts) > 1:
        suffix = col_name_parts[-1]
        if suffix in unique_vals:
            return True

    return False


def _get_labeled_numeric_data(df, col_names):
    """Get the labeled and numeric data columns from the DataFrame"""
    labeled_cols = []
    numeric_cols = []
    for col_name in col_names:
        col = df[col_name]
        if pd.api.types.is_numeric_dtype(col):
            if col.is_monotonic_increasing or col.is_monotonic_decreasing:
                labeled_cols.append(col_name)
            elif col.nunique() / col.size <= 0.05:
                labeled_cols.append(col_name)
            else:
                numeric_cols.append(col_name)

    # Return the names of the numeric features and the labeled features
    return numeric_cols, labeled_cols


def display_info_lists(data, list1_name, list1, list2_name, list2, list3_name, list3):
    # Display the lengths of the three lists
    print(f"Number of {list1_name} features {len(list1)},\n Number of {list2_name} features {len(list2)},\n Number of {list3_name} features {len(list3_name)}") 
    
    # Display a sample of the one-hot encoded features
    print(f"Samples of {list1_name} features")
    display(data[list1].sample(5))
    
    # Display a sample of the numeric features
    print(f"Samples of {list2_name} features")
    display(data[list3].sample(5))
    
    # Display a sample of the labeled features
    print(f"Samples of {list3_name} features")
    display(data[list3].sample(5))


def get_features_based_type(data, missing_features):
    # Get the names of the one-hot encoded features in natural language
    one_hot_natural = [col for col in missing_features if _is_one_hot_encoded(data[missing_features], col)]

    # Get the names of the numeric features
    numeric_features = list(set(one_hot_natural).symmetric_difference(set(missing_features)))
    numeric_features, labeled_features = _get_labeled_numeric_data(data, numeric_features)
    
    display_info_lists(data, "one-hot encoded", one_hot_natural, "numeric", numeric_features, "labeled", labeled_features)
    
    # Return the names of the one-hot encoded features and the numeric features
    return one_hot_natural, numeric_features, labeled_features

