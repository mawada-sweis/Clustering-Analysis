# import libraries
import pandas as pd
from sklearn.preprocessing import OneHotEncoder




def one_hot_encoding(data, column):

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
    # apply date transformation :
    # extract day, month and year from each date 
    # drop the original dates columns 
    # drop extracted year because it is one unique value = 2019

    # convert to pandas datetime 
    for column in columns:
        data[column] = pd.to_datetime(data[column])

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
    # do feature scaling using min max scaler
    scaler.fit(data[columns])
    scaled_data = scaler.transform(data[columns])
    data[columns] = scaled_data
    return data