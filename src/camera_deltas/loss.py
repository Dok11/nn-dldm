import tensorflow as tf


def custom_objective(y_true, y_pred):
    radian_to_meter_valuable = 5

    error = tf.math.square(y_pred - y_true)

    # x+y+z errors
    # trans_mag = tf.math.sqrt(error[0] + error[1] + error[2])
    # max_trans_err = max(error[0], error[1], error[2])

    # euler rotation xyz
    orient_mag = tf.math.sqrt(error[3] + error[4] + error[5])

    return tf.keras.backend.mean((radian_to_meter_valuable * orient_mag))


y_true = [[0.1, 0.2, 0.3, 0.05, 0.03, 0.07],
          [0.2, 0.3, 0.4, 0.06, 0.04, 0.08],
          [0.3, 0.4, 0.5, 0.07, 0.05, 0.09]]

y_pred = [[0.12, 0.21, 0.36, 0.054, 0.038, 0.073],
          [0.25, 0.33, 0.14, 0.066, 0.048, 0.089],
          [0.39, 0.46, 0.54, 0.071, 0.053, 0.094]]

print(custom_objective(y_true, y_pred))
