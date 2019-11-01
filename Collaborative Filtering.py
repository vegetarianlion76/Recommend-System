import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error


def predict(ratings, similarity, cf_type='user'):
    # User-CF
    if cf_type == 'user':
        # average bias
        mean_user_rating = ratings.mean(axis=1, keepdims=True)
        ratings_diff = (ratings - mean_user_rating)
        pred = mean_user_rating + similarity.dot(ratings_diff) / similarity.sum(axis=1, keepdims=True)

        # average score
        pred = similarity.dot(ratings) / similarity.sum(axis=1, keepdims=True)
    # Item-CF
    elif cf_type == 'item':
        pred = ratings.dot(similarity) / similarity.sum(axis=1)
    else:
        return None
    return pred


def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()]
    ground_truth = ground_truth[ground_truth.nonzero()]
    return np.sqrt(mean_squared_error(prediction, ground_truth))


# read data
header = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('u.data', sep='\t', names=header)
print(df.head())

# calculate the number of users and items
n_users = df.user_id.unique().shape[0]
n_items = df.item_id.unique().shape[0]

# train-test split
train_data, test_data = train_test_split(df, test_size=0.2)

# construct rating matrix
train_data_matrix = np.zeros(shape=[n_users, n_items])
for line in train_data.itertuples():
    # line: line_no user_id item_id rating timestamp
    train_data_matrix[line[1] - 1, line[2] - 1] = line[3] #the first indice is user_id, the second one is item_id

test_data_matrix = np.zeros(shape=(n_users, n_items))
for line in test_data.itertuples():
    test_data_matrix[line[1] - 1, line[2] - 1] = line[3]

# calculate similarity matrix
user_similarity = pairwise_distances(train_data_matrix, metric='cosine')
print(user_similarity)
item_similarity = pairwise_distances(train_data_matrix.T, metric='cosine')

# predict
user_prediction = predict(train_data_matrix, user_similarity, cf_type='user')
print(user_prediction)
item_prediction = predict(train_data_matrix, item_similarity, cf_type='item')

# calculate loss
print('User-CF RMSE: %.4f' % rmse(user_prediction, test_data_matrix))
print('Item-CF RMSE: %.4f' % rmse(item_prediction, test_data_matrix))
