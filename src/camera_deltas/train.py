import os

import shutil
import keras
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

def get_dataset():
    file_path = os.path.join(CURRENT_DIR, '..', '..', 'train-data', 'camera_deltas', 'data_000.npz')
    file_data = np.load(file_path)

    result = {
        'x1': file_data['x1'] / 255.,
        'x2': file_data['x2'] / 255.,
        'y': file_data['y'],
    }

    return result


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

BETA = 10


def custom_objective(y_true, y_pred):
    error = keras.backend.square(y_pred - y_true)
    trans_mag = keras.backend.sqrt(error[0] + error[1] + error[2])
    orient_mag = keras.backend.sqrt(error[3] + error[4] + error[5] + error[6])
    return keras.backend.mean(trans_mag + (BETA * orient_mag))


inputs = []
input_models = []

for input_idx in range(INPUT_NUMS):
    model_input = Input(shape=IMG_SHAPE)
    inputs.append(model_input)

    # 90x60 -> 23x15
    model = Conv2D(64, (11, 11), strides=4, input_shape=IMG_SHAPE, padding='same')(model_input)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = MaxPooling2D(pool_size=(2, 2))(model)
    model = Dropout(0.2)(model)

    # 23x15 -> 11x7
    model = Conv2D(96, (5, 5), padding='same')(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = MaxPooling2D(pool_size=(2, 2))(model)
    model = Dropout(0.2)(model)

    # 11x7 -> 5x3
    model = Conv2D(196, (3, 3), padding='same')(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = MaxPooling2D(pool_size=(2, 2))(model)
    model = Dropout(0.2)(model)

    input_models.append(model)

merged_layers = concatenate(input_models)

merged_layers = Flatten()(merged_layers)
merged_layers = Dense(512, activation='relu')(merged_layers)
merged_layers = Dropout(0.5)(merged_layers)

output = Dense(7, kernel_initializer='normal', activation='linear')(merged_layers)
model = Model(inputs=inputs, outputs=output)
model.compile(optimizer=Adam(0.0001, decay=0.00001), loss=custom_objective, metrics=['accuracy'])
model.summary()


# ============================================================================
# --- Train and print accuracy -----------------------------------------------
# ----------------------------------------------------------------------------

log_path = './logs'
callback = TensorBoard(log_path)
callback.set_model(model)
train_names = ['train_loss', 'train_accuracy']
val_names = ['val_loss', 'val_accuracy']

file_data = get_dataset()

train_and_valid_edge = 4000

# train
train_x1 = file_data['x1'][:train_and_valid_edge]
train_x2 = file_data['x2'][:train_and_valid_edge]
train_y = file_data['y'][:train_and_valid_edge]

# test
test_x1 = file_data['x1'][train_and_valid_edge:]
test_x2 = file_data['x2'][train_and_valid_edge:]
test_y = file_data['y'][train_and_valid_edge:]

# predict
idx_p = [0]
images_x1_p = test_x1[idx_p]
images_x2_p = test_x2[idx_p]
train_y_p = test_y[idx_p]

# train
for batch in range(1000000):
    idx = np.random.randint(0, len(train_x1), 64)
    images_x1 = train_x1[idx]
    images_x2 = train_x2[idx]
    images_y = train_y[idx]

    idx = np.random.randint(0, len(test_x1), 64)
    test_images_x1 = test_x1[idx]
    test_images_x2 = test_x2[idx]
    test_images_y = test_y[idx]

    logs = model.train_on_batch(x=[images_x1, images_x2], y=images_y)

    if batch % 100 == 0:
        # check model on the train data
        train_idx = np.random.randint(0, len(train_x1), 64)
        m_loss = model.test_on_batch(x=[train_x1[train_idx], train_x2[train_idx]], y=train_y[train_idx])

        # check model on the validation data
        valid_idx = np.random.randint(0, len(test_x1), 64)
        v_loss = model.test_on_batch(x=[test_x1[valid_idx], test_x2[valid_idx]], y=test_y[valid_idx])
        predict = model.predict(x=[images_x1_p, images_x2_p])

        print('%d [loss: %f, t.acc.: %.2f%%, v.acc.: %.2f%%]' % (batch, m_loss[0], m_loss[1], v_loss[1]))
        print('predict', predict[0])
        print('train  ', train_y_p[0])
        write_log(callback, train_names, logs, batch)
        write_log(callback, val_names, v_loss, batch)

        # save_models(model)
