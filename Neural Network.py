import os
import numpy as np
import tensorflow as tf

'''
np.random.seed(555)
tf.random.set_random_seed(555)
'''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def load_data():
    print('loading data ...')

    rating_file = 'u.data'
    rating_np = np.loadtxt(rating_file, dtype=np.int32)
    rating_np = rating_np[:, 0:3]  # remove timestamp
    rating_np[:, 0:2] = rating_np[:, 0:2] - 1
    n_user = np.max(rating_np[:, 0]) + 1
    n_item = np.max(rating_np[:, 1]) + 1
    train, val, test = dataset_split(rating_np)
    return n_user, n_item, train, val, test


def dataset_split(rating_np):
    print('splitting dataset ...')

    # train : validation : test = 6 : 2 : 2
    val_ratio = 0.2
    test_ratio = 0.2
    n_ratings = rating_np.shape[0]

    val_indices = np.random.choice(list(range(n_ratings)), size=int(n_ratings * val_ratio), replace=False)
    left = set(range(n_ratings)) - set(val_indices)
    test_indices = np.random.choice(list(left), size=int(n_ratings * test_ratio), replace=False)
    train_indices = list(left - set(test_indices))

    train = rating_np[train_indices]
    val = rating_np[val_indices]
    test = rating_np[test_indices]

    return train, val, test


def get_feed_dict(data, start, end):
    feed_dict = {user_indices: data[start:end, 0],
                 item_indices: data[start:end, 1],
                 ratings: data[start:end, 2]}
    return feed_dict


# === hyper-parameters === #
n_epochs = 20
batch_size = 8192
embed_dim = 2
hidden_dim = 4
l2_weight = 1e-8
lr = 0.01

# === load data === #
n_user, n_item, train_data, val_data, test_data = load_data()

# === define the model === #
# model input for a minibatch
user_indices = tf.placeholder(dtype=tf.int32, shape=[None], name='user_indices')
item_indices = tf.placeholder(dtype=tf.int32, shape=[None], name='item_indices')
ratings = tf.placeholder(dtype=tf.float32, shape=[None], name='ratings')

# user and item latent feature matrix
user_feature_matrix = tf.Variable(initial_value=tf.random.truncated_normal(shape=[n_user, embed_dim]),
                                  name='user_feature_matrix')
item_feature_matrix = tf.Variable(initial_value=tf.truncated_normal(shape=[n_item, embed_dim]),
                                  name='item_feature_matrix')

# user and item latent features for a minibatch
user_features = tf.nn.embedding_lookup(params=user_feature_matrix, ids=user_indices)
item_features = tf.nn.embedding_lookup(params=item_feature_matrix, ids=item_indices)

# concatenate user features and item features
concat = tf.concat([user_features, item_features], axis=-1)

# one hidden layer
W_1 = tf.Variable(initial_value=tf.random.truncated_normal(shape=[embed_dim * 2, hidden_dim], name='W_1'))
b_1 = tf.Variable(initial_value=tf.random.truncated_normal(shape=[hidden_dim], name='b_1'))
hidden = tf.nn.relu(tf.matmul(concat, W_1) + b_1)

# output layer
W_2 = tf.Variable(initial_value=tf.random.truncated_normal(shape=[hidden_dim, 1], name='W_2'))
b_2 = tf.Variable(initial_value=tf.random.truncated_normal(shape=[1], name='b_2'))
output = tf.matmul(hidden, W_2) + b_2

# loss
pred_loss = tf.reduce_mean(tf.square(ratings - output))
l2_loss = l2_weight * (
        tf.nn.l2_loss(user_features) + tf.nn.l2_loss(item_features) + tf.nn.l2_loss(W_1) + tf.nn.l2_loss(W_2))
loss = pred_loss + l2_loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)

# RMSE
rmse = tf.sqrt(pred_loss)

# === train === #
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(n_epochs):
        np.random.shuffle(train_data)
        start = 0
        while start < len(train_data):
            sess.run(optimizer, feed_dict=get_feed_dict(train_data, start, start + batch_size))
            start += batch_size

        train_loss, train_rmse = sess.run([loss, rmse], feed_dict=get_feed_dict(train_data, 0, len(train_data)))
        val_loss, val_rmse = sess.run([loss, rmse], feed_dict=get_feed_dict(val_data, 0, len(val_data)))
        test_loss, test_rmse = sess.run([loss, rmse], feed_dict=get_feed_dict(test_data, 0, len(test_data)))

        print('epoch %d    train loss: %.4f  rmse: %.4f    val loss: %.4f  rmse: %.4f    test loss: %.4f  rmse: %.4f'
              % (step, train_loss, train_rmse, val_loss, val_rmse, test_loss, test_rmse))
