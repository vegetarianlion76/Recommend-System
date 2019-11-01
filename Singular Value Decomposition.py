import pandas as pd
import numpy as np
from scipy.sparse import linalg
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()]
    ground_truth = ground_truth[ground_truth.nonzero()]
    return np.sqrt(mean_squared_error(prediction, ground_truth))


# read data
header = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('u.data', sep='\t', names=header)

# calculate the number of users and items
n_users = df.user_id.unique().shape[0]
n_items = df.item_id.unique().shape[0]

# train-test split
train_data, test_data = train_test_split(df, test_size=0.2)

# construct rating matrix
train_data_matrix = np.zeros(shape=(n_users, n_items))
for line in train_data.itertuples():
    train_data_matrix[line[1] - 1, line[2] - 1] = line[3]

test_data_matrix = np.zeros(shape=(n_users, n_items))
for line in test_data.itertuples():
    test_data_matrix[line[1] - 1, line[2] - 1] = line[3]

# SVD
u, s, vt = linalg.svds(train_data_matrix, k=20)
s = np.diag(s)
pred = np.dot(np.dot(u, s), vt)

print('SVD MSE: %.4f' % rmse(pred, test_data_matrix))
