import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics.pairwise import euclidean_distances
from dateutil import parser

city_encoder = joblib.load('model/city_encoder.sav')
country_encoder = joblib.load('model/country_encoder.sav')
normalizer = joblib.load('model/normalizer.sav')
normalized_df = pd.read_csv('model/normalized_df.csv')
dates_df = pd.read_csv(('model/dates_df.csv'))


def find_top_n_similar_with_weights(user_data, weights_dict, search_date, top_n=10, normalized_df=normalized_df):
    normalized_df_copy = normalized_df.drop(columns=['id'])
    weights_dict = dict(weights_dict)
    user_input = dict(user_data)
    
    # constants for slicing features
    distance_dims = 2
    tuition_dims = distance_dims + 1
    city_dims = tuition_dims + len(city_encoder.categories_[0])
    country_dims = city_dims + len(country_encoder.categories_[0])
    start_date_dims = country_dims + 1
    registration_expiration_dims = start_date_dims + 1
    """ 
    All preprocessed kinderarten data Part:
        1. Reading importance of features from user input (weights_dict)
        2. Initializing an array representing the importance of features (weights)
        3. Initializing matrix and diagonalizing it with the previously defined weights array (weighted_matrix)
        4. Appending start_date and registration_expiration cols to preprocessed normalized_df 
        5. Multiplying the normalized_df with the weighted_matrix to get a weighted_normalized_df (weighted_normalized_df)
    """
    # 1. initializing weights array representing the importance of each feature
    # columns in order: [latitude, longitude, tuition, cities, countries, start_date, registration_expiration]
    dims_numb = len(normalized_df_copy.columns) + 2

    # 2. Initializing an array representing the importance of features (weights)
    weights = np.zeros(dims_numb)
    weights[0:distance_dims] = weights_dict['location']
    weights[distance_dims:tuition_dims] = weights_dict['tuition']
    weights[tuition_dims:city_dims] = weights_dict['city']
    weights[city_dims:country_dims] = weights_dict['country']
    weights[country_dims:start_date_dims] = weights_dict['start_date']
    weights[start_date_dims:registration_expiration_dims] = weights_dict['registration_expiration']

    # 3. Initializing matrix and diagonalizing it with the previously defined weights array (weighted_matrix)
    weighted_matrix = np.zeros((dims_numb, dims_numb))
    np.fill_diagonal(weighted_matrix, weights)

    # 4. Appending start_date and registration_expiration cols to preprocessed normalized_df
    # Giving the kindergarten a higher rank if the start and registration did not start yet
    start_date_series = dates_df['start_date'].apply(
        lambda start_date: 1 if (parser.parse(start_date) > search_date) else 0)
    registration_expiration_series = dates_df['registration_expiration'].apply(
        lambda date: 1 if (parser.parse(date) > search_date) else 0)

    normalized_df_copy = pd.concat([normalized_df_copy, start_date_series], axis=1)
    normalized_df_copy = pd.concat([normalized_df_copy, registration_expiration_series], axis=1)

    weighted_normalized_df = np.matmul(weighted_matrix, normalized_df_copy.T).T

    """
    User input Part:
        1. Storing user input data in a dataframe (user_input_df)
        2. Encoding city and country columns in user_input
        3. Concatenating encoded city and country to user_input dataframe and dropping old cols
        4. Normalizing user input data (user_input_normalized)
        5. Appending 1's as `start_date` and `registration_expiration` cols to user_input_normalized 
        6. Multiplying weighted matrix by user input after normalizing it
    """

    # 1. storing user input data in a dataframe (user_input_df)
    user_input_df = pd.DataFrame(user_input, index=[0])

    # 2. encoding city and country columns in user_input
    user_input_city_encoded = city_encoder.transform(user_input_df[['city']]).toarray()
    user_input_country_encoded = country_encoder.transform(user_input_df[['country']]).toarray()

    # 3. concatenating encoded city and country to user_input dataframe
    user_input_df = pd.concat([user_input_df, pd.DataFrame(user_input_city_encoded, columns=city_encoder.categories_)],
                              axis=1)
    user_input_df = pd.concat(
        [user_input_df, pd.DataFrame(user_input_country_encoded, columns=country_encoder.categories_)], axis=1)

    user_input_df.drop(columns=['city', 'country'], inplace=True)

    # 4. scaling user_input
    user_input_normalized = normalizer.transform(user_input_df)

    # 5. appending `start_date` and `registration_expiration` features to user input and initializing values to 1's
    user_input_normalized = np.append(user_input_normalized[0], [1, 1], axis=0).reshape(1, user_input_normalized.shape[1] + 2)

    # 6. multiplying weighted matrix by user input after scaling it
    user_input_normalized_weighted = np.matmul(weighted_matrix, user_input_normalized.T).T

    """
    Similarity Part:
        1. Finding similarity between all pre-processed kindergartens and the processes user_input_df
        2. Sorting the result in descending order according to the similarity
        3. returning the Ids of the top N similar kindergarten to the user_input df
    """
    # 1. finding similarity between all pre-processed kindergartens and the processes user_input df
    # Euclidean Distances was the used metric to find distance between vectors
    similarity = pd.DataFrame(
        euclidean_distances(user_input_normalized_weighted, weighted_normalized_df).reshape(-1, 1),
        columns=['similarity'])
    similarity['id'] = normalized_df['id']

    # 2. sorting the result in descending order according to the similarity
    # sorting in ascending order because the more the euclidean distance is less the more the vector are similar
    # in case of using cosine distance metric, we will be sorting in descending order
    similarity_sorted = similarity.sort_values('similarity', ascending=True)

    # 3. returning the Ids of the top N similar kindergarten to the user_input df
    return similarity_sorted.iloc[:top_n].set_index('id').to_dict()
