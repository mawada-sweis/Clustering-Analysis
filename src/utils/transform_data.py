# import libraries
import pandas as pd
from sklearn.preprocessing import OneHotEncoder




def one_hot_encoding(data, column):

    """
    Applying One Hot Encoding 

    Args:
    data: dataframe 
    column: list of columns name to encode

    Return:
    new dataframe with encoded columns
    """
    # apply one hot encoding on categorical features
    encoder = OneHotEncoder()
    encoded = encoder.fit_transform(data[[column]])

    # get_feature_names_out() : return all columns name 
    # each contain the original column name + category value
    data[encoder.get_feature_names_out()] = encoded.toarray()

    # drop the original categorical column
    data.drop(column, inplace=True, axis=1)
    return data



def date_transform(data, columns):
    """
    Apply date transformation by extracting day, month and year from each date 

    Args: 
    data: dataframe
    columns: list of columns names to be transformes

    Return: 
    new dataframe
    """

    # convert to pandas datetime 
    for column in columns:
        data[column] = pd.to_datetime(data[column])

        # name new columns 
        day_column_name = column+'_day'
        month_column_name = column+'_month'
        year_column_name = column+'_year'

        # extract new features
        data[day_column_name] = data[column].dt.day
        data[month_column_name] = data[column].dt.month
        data[year_column_name] = data[column].dt.year
   
    # drop original date columns
    data.drop(columns= columns, inplace=True)

    return data



def scaling(data, columns, scaler):
    """
    Do feature scaling 
    
    Args:
    data: dataframe
    columns: list of columns names to be scaled
    scaler: the scaler to use

    Return: 
    new data frame
    """
    # fit and transform the data with scaler
    scaled_data = scaler.fit_transform(data[columns])

    # restore scaled data into the data frame 
    data[columns] = scaled_data
    
    return data