import os

import shutil
import tensorflow as tf
import numpy as np
from keras import Input, Model
from keras.callbacks import TensorBoard
from keras.layers import Conv2D, Dropout, BatchNormalization, Flatten, Dense, concatenate, Activation, \
    MaxPooling2D
from keras.optimizers import Adam

# ============================================================================
# --- GLOBAL PARAMS ----------------------------------------------------------
# ----------------------------------------------------------------------------

SIZE_X = 90
SIZE_Y = 60
INPUT_NUMS = 2
IMG_SHAPE = (SIZE_Y, SIZE_X, 1)
CURRENT_DIR: str = os.getcwd()
SAVED_MODEL: str = os.path.join(CURRENT_DIR, '..', '..', 'models', 'camera_deltas', 'model.h5')
SAVED_MODEL_W: str = os.path.join(CURRENT_DIR, '..', '..', 'models', 'camera_deltas', 'model_w.h5')


# ============================================================================
# --- Gets dataset with x1, x2 and result as `y` -----------------------------
# ----------------------------------------------------------------------------

def column(matrix, i):
    return np.array([row[i] for row in matrix])


def get_dataset():
    file_name = 'data_' + str(SIZE_X) + 'x' + str(SIZE_Y) + '_000.npz'
    file_path = os.path.join(CURRENT_DIR, '..', '..', 'train-data', 'camera_deltas', file_name)
    file_data = np.load(file_path, allow_pickle=True)

    return (
        column(file_data['train'], 0) / 255.,
        column(file_data['train'], 1) / 255.,
        column(file_data['train'], 2),
        column(file_data['valid'], 0) / 255.,
        column(file_data['valid'], 1) / 255.,
        column(file_data['valid'], 2),
    )


# tensorboard --logdir=./logs --host=127.0.0.1
shutil.rmtree(os.path.join(CURRENT_DIR, 'logs'), ignore_errors=True)


def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()


def save_models(model_for_save):
    model_for_save.save(SAVED_MODEL)
    model_for_save.save_weights(SAVED_MODEL_W)


# ============================================================================
# --- Construct neural network -----------------------------------------------
# ----------------------------------------------------------------------------

def loss_in_cm(y_true, y_pred):
    y_true *= 100
    y_pred *= 100

    return tf.math.sqrt(tf.math.pow(y_pred[0] - y_true[0], 2) +
                        tf.math.pow(y_pred[1] - y_true[1], 2) +
                        tf.math.pow(y_pred[2] - y_true[2], 2))


# Currently just summarize all errors
def loss_in_radian(y_true, y_pred):
    error = tf.math.square(y_pred - y_true)
    return error[3] + error[4] + error[5] + error[6]


def custom_objective(y_true, y_pred):
    radian_to_meter_valuable = 5

    error = tf.math.square(y_pred - y_true)

    trans_mag = tf.math.sqrt(error[0] + error[1] + error[2])  # x+y+z errors
    orient_mag = tf.math.sqrt(error[3] + error[4] + error[5] + error[6])  # w+x+y+z errors

    return tf.keras.backend.mean(trans_mag + (radian_to_meter_valuable * orient_mag))


inputs = []
input_models = []

for input_idx in range(INPUT_NUMS):
    model_input = Input(shape=IMG_SHAPE)
    inputs.append(model_input)

    # 90x60 -> 28x18
    model = Conv2D(128, (7, 7), strides=3, input_shape=IMG_SHAPE, padding='valid')(model_input)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)

    # 28x18 -> 14x9
    model = MaxPooling2D(pool_size=(2, 2))(model)
    model = Dropout(0.35)(model)

    # 14x9 -> 7x5
    model = Conv2D(256, (5, 5), strides=2, padding='same')(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)

    # 7x5 -> 3x2
    model = MaxPooling2D(pool_size=(2, 2))(model)
    model = Dropout(0.35)(model)

    # 3x2 -> 3x2
    model = Conv2D(512, (3, 3), padding='same')(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)

    # 3x2 -> 1x1
    model = MaxPooling2D(pool_size=(2, 2))(model)
    model = Dropout(0.35)(model)

    input_models.append(model)

merged_layers = concatenate(input_models)

merged_layers = Flatten()(merged_layers)
merged_layers = Dense(1024, activation='relu')(merged_layers)
merged_layers = Dense(2048, activation='relu')(merged_layers)
merged_layers = Dropout(0.35)(merged_layers)

output = Dense(7, kernel_initializer='normal', activation='linear')(merged_layers)
model = Model(inputs=inputs, outputs=output)
model.compile(optimizer=Adam(0.0001, decay=0.00001),
              loss=custom_objective,
              metrics=[loss_in_cm, loss_in_radian])

model.summary()

if os.path.isfile(SAVED_MODEL_W):
    model.load_weights(SAVED_MODEL_W)
    print('weights are loaded')


# ============================================================================
# --- Train and print accuracy -----------------------------------------------
# ----------------------------------------------------------------------------

callback = TensorBoard('./logs')
callback.set_model(model)
train_names = ['train_loss', 'train_loss_in_cm', 'train_loss_in_radian']
val_names = ['val_loss', 'val_loss_in_cm', 'val_loss_in_radian']

(train_x1, train_x2, train_y, test_x1, test_x2, test_y) = get_dataset()
train_batch_size = 512

# predict
idx_p = [0]
images_x1_p = test_x1[idx_p]
images_x2_p = test_x2[idx_p]
train_y_p = test_y[idx_p]

# train
for batch in range(9000000):
    idx = np.random.randint(0, len(train_x1), train_batch_size)
    images_x1 = train_x1[idx]
    images_x2 = train_x2[idx]
    images_y = train_y[idx]

    logs = model.train_on_batch(x=[images_x1, images_x2], y=images_y)

    if batch % 200 == 0 and batch > 0:
        # check model on the validation data
        valid_idx = np.random.randint(0, len(test_x1), train_batch_size)
        v_loss = model.test_on_batch(x=[test_x1[valid_idx], test_x2[valid_idx]], y=test_y[valid_idx])

        print('%d [loss: %f, t.loss.sm.: %.2f, v.loss.sm.: %.2f]' % (batch, logs[0], logs[1], v_loss[1]))
        write_log(callback, train_names, logs, batch)
        write_log(callback, val_names, v_loss, batch)

        save_models(model)
