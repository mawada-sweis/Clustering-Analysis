# import libraries
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler



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



def date_transform(data):
    # apply date transformation :
    # extract day, month and year from each date 
    # drop the original dates columns 
    # drop extracted year because it is one unique value = 2019

    # convert to pandas datetime 
    data['first_travel_date'] = pd.to_datetime(data['first_travel_date'])
    data['last_travel_date'] = pd.to_datetime(data['last_travel_date'])

    # extract new features
    data['first_travel_day'] = data['first_travel_date'].dt.day
    data['first_travel_month'] = data['first_travel_date'].dt.month
    data['first_travel_year'] = data['first_travel_date'].dt.year

    data['last_travel_day'] = data['last_travel_date'].dt.day
    data['last_travel_month'] = data['last_travel_date'].dt.month
    data['last_travel_year'] = data['last_travel_date'].dt.year
    
    # drop uneeded columns
    data.drop(columns= ['first_travel_date', 'last_travel_date', \
        'first_travel_year', 'last_travel_year'], inplace=True)

    return data



def scaling(data, columns):
    # do feature scaling using min max scaler
    scaler = MinMaxScaler()
    scaler.fit(data[columns])
    scaled_data = scaler.transform(data[columns])
    data[columns] = scaled_data
    return data