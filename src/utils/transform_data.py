# import libraries
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def one_hot_encoding(data: pd.DataFrame, categorical_features:list) -> pd.DataFrame:
    """Apply one hot encoding to categorical features in a dataframe.

    Args:
        data (pd.DataFrame): Input dataframe.
        categorical_features (list): List of column names to be encoded.

    Returns:
        pd.DataFrame: A new dataframe with encoded columns.

    """
    # Instantiate the OneHotEncoder
    encoder = OneHotEncoder()

    # Apply one hot encoding to categorical features
    encoded = encoder.fit_transform(data[categorical_features])

    # Get the names of the encoded features
    encoded_feature_names = encoder.get_feature_names_out()

    # Create a new dataframe for the encoded features
    encoded_df = pd.DataFrame(encoded.toarray(), columns=encoded_feature_names)

    return pd.concat([data.drop(columns=categorical_features), encoded_df], axis=1)



def date_transform(data: pd.DataFrame, date_columns:list) -> pd.DataFrame:
    """Extract day, month, and year from date columns in a dataframe.

    Args: 
        data (pd.DataFrame): Input dataframe.
        date_columns (list): List of column names to be transformed.

    Returns: 
        pd.DataFrame: A new dataframe with transformed columns.

    """
    # Convert date columns to pandas datetime and extract new features
    for column in date_columns:
        data[column] = pd.to_datetime(data[column])
        data[column+'_day'] = data[column].dt.day
        data[column+'_month'] = data[column].dt.month
        data[column+'_year'] = data[column].dt.year
   
    # Drop original date columns
    data.drop(columns=date_columns, inplace=True)

    return data


def scaling(data: pd.DataFrame, numeric_columns: list, scaler) -> pd.DataFrame:
    """Apply feature scaling to numeric columns in a dataframe.

    Args:
        data (pd.DataFrame): Input dataframe.
        numeric_columns (list): List of column names to be scaled.
        scaler: The scaler to use.

    Returns: 
        pd.DataFrame: A new dataframe with scaled columns.

    """
    # Fit and transform the data with the scaler
    scaled_data = scaler.fit_transform(data[numeric_columns])

    # Restore scaled data into the new dataframe 
    data[numeric_columns] = scaled_data
    
    return data
