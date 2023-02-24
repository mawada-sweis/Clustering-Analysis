import pandas as pd
from IPython.display import display


def delete_columns(df: pd.DataFrame, columns_to_delete: list) -> None:
    """Deletes all columns mentioned in the given list.

    Args:
        df (pd.DataFrame): Current dataframe.
        columns_to_delete (list): List of column names to delete.
    """
    # Print the number of columns in the current dataset
    print(f"Current number of columns: {df.shape[1]}")

    # Drop columns from the dataset
    df.drop(columns_to_delete, axis="columns", inplace=True)

    # Print the number of columns in the updated dataset
    print(f"Updated number of columns: {df.shape[1]}")


def filter_direct_missing_columns(df: pd.DataFrame, lower_threshold=None, upper_threshold=None) -> list:
    """Filters a pandas DataFrame to return a list of the columns that have missing
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
    # Get the percentage of missing values for each column
    missing_percentage = df['missing_percentage']
    
    # Filter the DataFrame based on the user-provided thresholds
    if lower_threshold is None:
        filtered_df = missing_percentage[missing_percentage > upper_threshold]
    elif upper_threshold is None:
        filtered_df = missing_percentage[missing_percentage < lower_threshold]
    else:
        filtered_df = missing_percentage[
            (missing_percentage > lower_threshold) & 
            (missing_percentage < upper_threshold)
        ]

    # Return a list of the column names in the filtered DataFrame
    return list(filtered_df.index)


def fill_by_mode(df: pd.DataFrame, columns_name: str) -> str:
    """Fill missing values in a pandas DataFrame columns with the mode value.
    
    Args:
        df (pd.DataFrame): A pandas DataFrame with the column to fill.
        columns_name (str): The name of the columns to fill
    
    Returns:
        str: A string indicating the success of the operation.
    """
    # Fill missing values in the specified column using the mode
    mode_value = df[columns_name].mode().iloc[0,0]
    df[columns_name] = df[columns_name].fillna(mode_value)
    
    # Display a success message
    print(f"Provided columns '{columns_name}' successfully filled by mode value.")


def filter_indirect_missing_columns(df: pd.DataFrame, columns_to_filter: list, lower_threshold=None, upper_threshold=None) -> list:
    """Filter a pandas DataFrame based on the percentage of indirect missing values for a list of columns.

    Args:
        df (pd.DataFrame): A pandas DataFrame.
        columns_to_filter (list): A list of columns name to filter.
        lower_threshold (float): The minimum percentage of indirect missing values allowed for columns to be included in the output. Default is None.
        upper_threshold (float): The maximum percentage of indirect missing values allowed for columns to be included in the output. Default is None.

    Returns:
        list: A list of columns name in the filtered DataFrame
    """
    # Filter the Dataframe columns
    data = df[columns_to_filter]
    
    # Get the maximum and minimum value of filtered data
    max_value = data.max(axis=1)
    min_value = data.min(axis=1)
    
    # Convert columns in the DataFrame to numeric data type
    filtered_df = data.apply(pd.to_numeric, errors='coerce')

    # Filter the DataFrame based on the user-provided thresholds for each column
    if lower_threshold is None:
        filtered_df = df.loc[max_value > upper_threshold]

    elif upper_threshold is None:
        filtered_df = df.loc[min_value < lower_threshold]
    
    else:
        filtered_df = df.loc[
            (max_value > lower_threshold) & 
            (min_value < upper_threshold)
        ]

    # Return a list of the column names in the filtered DataFrame
    return list(filtered_df.index)


def is_one_hot_encoded(df: pd.DataFrame, col_name:str) -> bool:
    """Check if the given column is one-hot encoded in the DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame to check.
        col_name (str): The name of the column to check.

    Returns:
        bool: True if the column is one-hot encoded, False otherwise.
    """
    # Get unique values in the column
    unique_vals = df[col_name].unique()
    
    # Create set of all possible values to check against
    check_num = (0, 1, 999, 997, -9998, 995, -9999)

    # Return True if the column is one-hot encoded
    # by determine whether check_num is a subset of unique_vals
    return set(check_num).issuperset(set(unique_vals))


def _get_labeled_numeric_data(df: pd.DataFrame, col_names: list) -> tuple:
    """Extracts labeled and numeric data from a DataFrame for a list of specified columns.

    Args:
        df (pd.DataFrame): A pandas DataFrame.
        col_names (list): A list of column names to extract labeled and numeric data from.

    Returns:
        tuple: A tuple containing a list of the names of the numeric columns and a list of the names of the labeled columns.
    """
    # Initialize empty lists to store labeled and numeric columns
    labeled_cols = []
    numeric_cols = []
    
    # Loop through the specified column names
    for col_name in col_names:
        col = df[col_name]
        
        # Check if the column is numeric and not one-hot encoded
        if pd.api.types.is_numeric_dtype(col) and not is_one_hot_encoded(df, col_name):

            # Check if the column is labeled
            if col.is_monotonic_increasing or col.is_monotonic_decreasing or col.dtype == 'int64':
                labeled_cols.append(col_name)
            
            # Otherwise, add it to the list of numeric columns
            elif col.dtype == 'float':
                numeric_cols.append(col_name)

    # Return the names of the numeric columns and the labeled columns as a tuple
    return numeric_cols, labeled_cols


def display_info_lists(data: pd.DataFrame, *args) -> None:
    """
    Displays information about lists of features in a pandas DataFrame.

    Args:
        data (pd.DataFrame): The pandas DataFrame to display information for.
        *args: Any number of tuples of the form (features_type, features)
            features_type (str): A string representing the type of features.
            features (list): A list of feature names that belong to the DataFrame.
    """
    for features_type, features in args:
        
        # Display the number of features in the list
        num_features = len(features)
        print(f"Number of {features_type} features: {num_features}")
        
        # Display a sample of the features if there are any
        if num_features > 0:
            sample_size = 5
            sample = data[features].sample(sample_size)
            print(f"Samples of {features_type} features (sample size={sample_size}):")
            display(sample)


def get_features_based_type(df: pd.DataFrame, missing_features: list) -> tuple:
    """This function takes a pandas DataFrame and a list of missing features and 
    returns a tuple of three lists of feature names.
    
    The first list contains the names of one-hot encoded features in natural. 
    The second list contains the names of numeric features.
    The third list contains the names of labeled features, which are categorical features.
    
    Args:
        df (pd.DataFrame): A pandas DataFrame containing the data.
        missing_features (list): A list of missing features in the DataFrame.

    Returns:
        tuple: A tuple of three lists of feature names: one-hot encoded features, numeric features, and labeled features.
    """
    # Get the names of the one-hot encoded features in natural
    one_hot_natural = [col for col in missing_features if is_one_hot_encoded(df[missing_features], col)]

    # Get the names of the numeric features
    numeric_features = list(set(one_hot_natural).symmetric_difference(set(missing_features)))
    numeric_features, labeled_features = _get_labeled_numeric_data(df, numeric_features)
    
    # Display information about different sets of features
    display_info_lists(df, 
                        ("one-hot encoded", one_hot_natural),
                        ("numeric", numeric_features),
                        ("labeled", labeled_features))
    
    # Return the names of the one-hot encoded features and the numeric features
    return one_hot_natural, numeric_features, labeled_features


def fill_numeric(df:pd.DataFrame, numeric_features:list, missing_types_label:list) -> None:
    """Fill missing numeric values in a pandas DataFrame.

    Args:
        df (pd.DataFrame): Input pandas DataFrame.
        numeric_features (list): List of numeric features to fill.
        missing_types_label (list): List of missing types.
    """
    # Check if the missing type is 995 and replace with 0
    if 995 in missing_types_label:
        df[numeric_features] = df[numeric_features].replace(995, 0)
    
    # Check if the missing type is -9998 and replace with mode value
    if -9998 in missing_types_label:
        mode_value = df[numeric_features].mode().iloc[0][0]
        df[numeric_features] = df[numeric_features].replace(-9998, mode_value)
    
    # If the missing type is not recognized, raise an error message
    else: 
        raise ValueError("Invalid missing type.")
    
    print("Numeric values have been filled successfully.")


def grouping_encoded_features(prefixes:list, features_names:list) -> dict:
    """Groups encoded features by their prefixes.

    Args:
        prefixes (list): A list of prefixes for the encoded features.
        feature_names (list): A list of all previously encoded feature names.

    Returns:
        dict: A dictionary mapping each prefix to a list of feature names with that prefix.
    """
    return {
        prefix: [col for col in features_names if col.startswith(prefix)]
        for prefix in prefixes
    }


def handle_missing_codes_encoded_features(data: pd.DataFrame, missing_codes: list, encoded_features: dict) -> pd.DataFrame:
    """Handles missing codes in encoded features by creating new columns for missing codes and replacing missing codes with 0.

    Args:
        data (pd.DataFrame): Input pandas DataFrame.
        missing_codes (list): List of codes to replace their values and create new columns for them.
        encoded_features (dict): Dictionary mapping each prefix to a list of feature names with that prefix.

    Returns:
        A modified pandas DataFrame.
    """
    for prefix, features in encoded_features.items():
        for code in missing_codes:
            new_col_name = f"{prefix}{code}"
            data[new_col_name] = (data[features[0]] == code).astype(int)
            data.loc[data[features[0]] == code, features] = 0

    return data


def convert_missing_to_label(data: pd.DataFrame, ordinal_features:list, missing_codes:list, new_labels:list) -> pd.DataFrame:
    """replace missing codes to label for ordinal features in a dataframe

    Args:
        data (pd.DataFrame): Input dataframe.
        ordinal_features (list): List of column names to be encoded.
        missing_codes (list): list of missing codes to fit
        new_label_for_missing (list): list of new values to replace missing codes with. 

    Returns:
        pd.DataFrame: A new dataframe with new labels for missing.
    """

    data[ordinal_features] = data[ordinal_features].replace(to_replace = missing_codes, value = new_labels)
    return data