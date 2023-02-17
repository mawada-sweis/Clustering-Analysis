import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler


def one_hot_encoding(data, column):
    encoder = OneHotEncoder()
    encoded = encoder.fit_transform(data[[column]])
    data[encoder.get_feature_names_out()] = encoded.toarray()
    data.drop(column, inplace=True, axis=1)
    return data



def date_transform(data):
    data['first_travel_date'] = pd.to_datetime(data['first_travel_date'])
    data['last_travel_date'] = pd.to_datetime(data['last_travel_date'])

    data['first_travel_day'] = data['first_travel_date'].dt.day
    data['first_travel_month'] = data['first_travel_date'].dt.month
    data['first_travel_year'] = data['first_travel_date'].dt.year

    data['last_travel_day'] = data['last_travel_date'].dt.day
    data['last_travel_month'] = data['last_travel_date'].dt.month
    data['last_travel_year'] = data['last_travel_date'].dt.year
    
    data.drop(columns= ['first_travel_date', 'last_travel_date', 'first_travel_year', 'last_travel_year'], inplace=True)
    return data


def scaling(data, columns):
    scaler = MinMaxScaler()
    scaler.fit(data[columns])
    scaled_data = scaler.transform(data[columns])
    data[columns] = scaled_data
    return data