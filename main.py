from __future__ import annotations

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

from recsysNN_utils import load_data


def build_and_train():
    item_train, user_train, y_train, item_features, user_features, item_vecs, movie_dict, user_to_genre = load_data()

    num_user_features = user_train.shape[1] - 3
    num_item_features = item_train.shape[1] - 1
    u_s = 3
    i_s = 1

    item_scaler = StandardScaler()
    user_scaler = StandardScaler()
    y_scaler = MinMaxScaler(feature_range=(-1, 1))

    item_scaled = item_scaler.fit_transform(item_train)
    user_scaled = user_scaler.fit_transform(user_train)
    y_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1))

    X_item_tr, X_item_te = train_test_split(item_scaled, train_size=0.8, shuffle=True, random_state=1)
    X_user_tr, X_user_te = train_test_split(user_scaled, train_size=0.8, shuffle=True, random_state=1)
    Y_tr, Y_te = train_test_split(y_scaled, train_size=0.8, shuffle=True, random_state=1)

    num_outputs = 32
    tf.random.set_seed(1)

    user_NN = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(num_user_features,)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_outputs, activation='linear'),
    ])

    item_NN = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(num_item_features,)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_outputs, activation='linear'),
    ])

    input_user = tf.keras.layers.Input(shape=(num_user_features,))
    vu = user_NN(input_user)
    vu = tf.linalg.l2_normalize(vu, axis=1)

    input_item = tf.keras.layers.Input(shape=(num_item_features,))
    vm = item_NN(input_item)
    vm = tf.linalg.l2_normalize(vm, axis=1)

    output = tf.keras.layers.Dot(axes=1)([vu, vm])
    model = tf.keras.Model([input_user, input_item], output)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                  loss=tf.keras.losses.MeanSquaredError())

    model.fit([X_user_tr[:, u_s:], X_item_tr[:, i_s:]], Y_tr, epochs=5, batch_size=1024,
              validation_data=([X_user_te[:, u_s:], X_item_te[:, i_s:]], Y_te), verbose=1)

    val_mse = float(model.evaluate([X_user_te[:, u_s:], X_item_te[:, i_s:]], Y_te, verbose=0))
    print("Validation MSE:", val_mse)


if __name__ == "__main__":
    build_and_train()

