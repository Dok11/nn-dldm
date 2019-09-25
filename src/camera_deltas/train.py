import os
import shutil

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.callbacks_v1 import TensorBoard
from tensorflow.python.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Dropout, concatenate, \
    Flatten, Dense

# ============================================================================
# --- GLOBAL PARAMS ----------------------------------------------------------
# ----------------------------------------------------------------------------

SIZE_X = 90
SIZE_Y = 60
IMG_SHAPE = (SIZE_Y, SIZE_X, 1)
CURRENT_DIR: str = os.getcwd()
SAVED_MODEL: str = os.path.join(CURRENT_DIR, '..', '..', 'models', 'camera_deltas', 'model.h5')
SAVED_MODEL_W: str = os.path.join(CURRENT_DIR, '..', '..', 'models', 'camera_deltas', 'model_w.h5')

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


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
        column(file_data['train'], 0),  # source image path
        column(file_data['train'], 1),  # destination
        column(file_data['train'], 2),  # fov
        column(file_data['train'], 3),  # result
        column(file_data['valid'], 0),  # source image path
        column(file_data['valid'], 1),  # destination
        column(file_data['valid'], 2),  # fov
        column(file_data['valid'], 3),  # result
        file_data['images'] / 255.,
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

    # x+y+z errors
    trans_mag = tf.math.sqrt(error[0] + error[1] + error[2])
    # max_trans_err = max(error[0], error[1], error[2])

    # euler rotation xyz
    orient_mag = tf.math.sqrt(error[3] + error[4] + error[5])

    return tf.keras.backend.mean(trans_mag + (radian_to_meter_valuable * orient_mag))


def get_image_branch():
    shared_input = Input(IMG_SHAPE)

    # 90x60 -> 28x18
    shared_layer = Conv2D(128, (7, 7), strides=3, input_shape=IMG_SHAPE, padding='valid')(shared_input)
    shared_layer = BatchNormalization()(shared_layer)
    shared_layer = Activation('relu')(shared_layer)

    # 28x18 -> 14x9
    shared_layer = MaxPooling2D(pool_size=(2, 2))(shared_layer)
    shared_layer = Dropout(0.35)(shared_layer)

    # 14x9 -> 7x5
    shared_layer = Conv2D(256, (5, 5), strides=2, padding='same')(shared_layer)
    shared_layer = BatchNormalization()(shared_layer)
    shared_layer = Activation('relu')(shared_layer)

    # 7x5 -> 3x2
    shared_layer = MaxPooling2D(pool_size=(2, 2))(shared_layer)
    shared_layer = Dropout(0.35)(shared_layer)

    # 3x2 -> 3x2
    shared_layer = Conv2D(512, (3, 3), padding='same')(shared_layer)
    shared_layer = BatchNormalization()(shared_layer)
    shared_layer = Activation('relu')(shared_layer)

    # 3x2 -> 1x1
    shared_layer = MaxPooling2D(pool_size=(2, 2))(shared_layer)
    shared_layer = Dropout(0.35)(shared_layer)

    return Model(shared_input, shared_layer, name='shared_model')


shared_model = get_image_branch()
shared_model.summary()

image_a = Input(IMG_SHAPE)
image_b = Input(IMG_SHAPE)

branch_a = shared_model(image_a)
branch_b = shared_model(image_b)

merged_layers = concatenate([branch_a, branch_b], axis=-1)

merged_layers = Flatten()(merged_layers)
merged_layers = Dense(1024, activation='relu')(merged_layers)
merged_layers = Dense(1024, activation='relu')(merged_layers)
merged_layers = Dropout(0.35)(merged_layers)

output = Dense(6, kernel_initializer='normal', activation='linear')(merged_layers)
model = Model(inputs=[image_a, image_b], outputs=output)
model.compile(optimizer=tf.keras.optimizers.Adam(0.00005, decay=0.00001),
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

(train_x1, train_x2, train_fov, train_y,
 test_x1, test_x2, test_fov, test_y,
 images) = get_dataset()
train_batch_size = 64


# train
for batch in range(9000000):
    idx = np.random.randint(0, len(train_x1), train_batch_size)
    images_idx_x1 = train_x1[idx]
    images_idx_x2 = train_x2[idx]
    images_x1 = images[images_idx_x1]
    images_x2 = images[images_idx_x2]
    result = train_y[idx]

    logs = model.train_on_batch(x=[images_x1, images_x2], y=result)

    if batch % 200 == 0 and batch > 0:
        # check model on the validation data
        valid_idx = np.random.randint(0, len(test_x1), train_batch_size)
        valid_images_idx_x1 = test_x1[valid_idx]
        valid_images_idx_x2 = test_x2[valid_idx]
        valid_images_x1 = images[valid_images_idx_x1]
        valid_images_x2 = images[valid_images_idx_x2]
        valid_result = test_y[valid_idx]

        v_loss = model.test_on_batch(x=[valid_images_x1, valid_images_x2], y=valid_result)

        print('%d [loss: %f]' % (batch, logs[0]))
        write_log(callback, train_names, logs, batch)
        write_log(callback, val_names, v_loss, batch)

        save_models(model)
