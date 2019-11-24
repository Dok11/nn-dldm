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

SIZE_X = 224
SIZE_Y = 224
IMG_SHAPE = (SIZE_Y, SIZE_X, 3)
CURRENT_DIR: str = os.getcwd()
SAVED_MODEL: str = os.path.join(CURRENT_DIR, '..', '..', 'models', 'camera_deltas', 'model.h5')
SAVED_MODEL_W: str = os.path.join(CURRENT_DIR, '..', '..', 'models', 'camera_deltas', 'model_w.h5')

# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


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
        file_data['images'],
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
    delta = y_pred - y_true
    x = delta[0] * 100
    y = delta[4] * 100
    z = delta[8] * 100

    return tf.sqrt(tf.pow(x, 2) +
                   tf.pow(y, 2) +
                   tf.pow(z, 2))


def loss_in_cm_x(y_true, y_pred):
    delta = y_pred - y_true
    return tf.sqrt(tf.pow(delta[0] * 100, 2))


def loss_in_cm_y(y_true, y_pred):
    delta = y_pred - y_true
    return tf.sqrt(tf.pow(delta[4] * 100, 2))


def loss_in_cm_z(y_true, y_pred):
    delta = y_pred - y_true
    return tf.sqrt(tf.pow(delta[8] * 100, 2))


# Currently just summarize all errors
def loss_in_radian(y_true, y_pred):
    error = y_pred - y_true
    return tf.math.abs(error[9] + error[10] + error[11])


def custom_objective(y_true, y_pred):
    radian_to_meter_valuable = 5

    error = tf.math.square(y_pred - y_true)
    # error = tf.Print(error, [error], 'error', summarize=1000)

    # x+y+z errors
    # trans_error = tf.math.sqrt(error[0] + error[4] + error[8])
    trans_error = tf.math.sqrt(error[0] + error[1]*0 + error[2]*0 + error[3]*0 + error[4] + error[5]*0 + error[6]*0 + error[7]*0 + error[8])
    # trans_error = error[0] + error[1] + error[2] + error[3] + error[4] + error[5] + error[6] + error[7] + error[8]

    print(trans_error)

    # euler rotation xyz
    orient_error = tf.math.sqrt(error[9] + error[10] + error[11])

    return tf.reduce_mean(trans_error + (radian_to_meter_valuable * orient_error), axis=-1)


def get_image_branch():
    shared_input = Input(IMG_SHAPE)

    # 224x224 -> 74x74
    shared_layer = Conv2D(50, (5, 5), strides=3, input_shape=IMG_SHAPE, padding='valid')(shared_input)
    shared_layer = BatchNormalization()(shared_layer)
    shared_layer = Activation('selu')(shared_layer)

    # 74x74 -> 37x37
    shared_layer = MaxPooling2D(pool_size=(2, 2))(shared_layer)
    shared_layer = Dropout(0.35)(shared_layer)

    # 37x37 -> 37x37
    shared_layer = Conv2D(100, (4, 4), padding='same')(shared_layer)
    shared_layer = BatchNormalization()(shared_layer)
    shared_layer = Activation('selu')(shared_layer)

    # 37x37 -> 18x18
    shared_layer = MaxPooling2D(pool_size=(2, 2))(shared_layer)
    shared_layer = Dropout(0.35)(shared_layer)

    # 18x18 -> 18x18
    shared_layer = Conv2D(200, (3, 3), padding='same')(shared_layer)
    shared_layer = BatchNormalization()(shared_layer)
    shared_layer = Activation('selu')(shared_layer)

    # 18x18 -> 9x9
    shared_layer = MaxPooling2D(pool_size=(2, 2))(shared_layer)
    shared_layer = Dropout(0.35)(shared_layer)

    # 9x9 -> 9x9
    shared_layer = Conv2D(400, (3, 3), padding='same')(shared_layer)
    shared_layer = BatchNormalization()(shared_layer)
    shared_layer = Activation('selu')(shared_layer)

    # 9x9 -> 4x4
    shared_layer = MaxPooling2D(pool_size=(2, 2))(shared_layer)
    shared_layer = Dropout(0.35)(shared_layer)

    return Model(shared_input, shared_layer, name='shared_model')


shared_model = get_image_branch()
shared_model.summary()

image_a = Input(IMG_SHAPE)
image_b = Input(IMG_SHAPE)
image_fov = Input((1,))

branch_a = shared_model(image_a)
branch_b = shared_model(image_b)

merged_layers = concatenate([branch_a, branch_b])

merged_layers = Flatten()(merged_layers)
merged_layers = concatenate([merged_layers, image_fov])
merged_layers = Dense(1024, activation='selu')(merged_layers)
merged_layers = Dense(1024, activation='selu')(merged_layers)

output = Dense(12, kernel_initializer='normal', activation='linear')(merged_layers)
model = Model(inputs=[image_a, image_b, image_fov], outputs=output)
model.compile(optimizer=tf.keras.optimizers.Adam(0.00005, decay=0.00001),
              loss=custom_objective,
              metrics=[loss_in_cm, loss_in_radian, loss_in_cm_x, loss_in_cm_y, loss_in_cm_z])

model.summary()

if os.path.isfile(SAVED_MODEL_W):
    model.load_weights(SAVED_MODEL_W)
    print('weights are loaded')


# ============================================================================
# --- Train and print accuracy -----------------------------------------------
# ----------------------------------------------------------------------------

callback = TensorBoard('./logs')
callback.set_model(model)
train_names = ['train_loss', 'train_loss_in_cm', 'train_loss_in_radian', 'train_loss_in_cm_x', 'train_loss_in_cm_y', 'train_loss_in_cm_z']
val_names = ['val_loss', 'val_loss_in_cm', 'val_loss_in_radian', 'val_loss_in_cm_x', 'val_loss_in_cm_y', 'val_loss_in_cm_z']

(train_x1, train_x2, train_fov, train_y,
 test_x1, test_x2, test_fov, test_y,
 images) = get_dataset()
train_batch_size = 64


# train
sum_logs = []
for batch in range(50000001):
    idx = np.random.randint(0, len(train_x1), train_batch_size)
    images_idx_x1 = train_x1[idx]
    images_idx_x2 = train_x2[idx]
    images_x1 = images[images_idx_x1] / 255.
    images_x2 = images[images_idx_x2] / 255.
    images_fov = train_fov[idx]
    result = train_y[idx]

    logs = model.train_on_batch(x=[images_x1, images_x2, images_fov], y=result)
    sum_logs.append(logs)

    if batch % 200 == 0 and batch > 0:
        # check model on the validation data
        valid_idx = np.random.randint(0, len(test_x1), train_batch_size)
        valid_images_idx_x1 = test_x1[valid_idx]
        valid_images_idx_x2 = test_x2[valid_idx]
        valid_images_x1 = images[valid_images_idx_x1] / 255.
        valid_images_x2 = images[valid_images_idx_x2] / 255.
        valid_images_fov = train_fov[valid_idx]
        valid_result = test_y[valid_idx]

        v_loss = model.test_on_batch(x=[valid_images_x1, valid_images_x2, valid_images_fov], y=valid_result)

        avg_logs = np.average(sum_logs, axis=0)
        sum_logs = []

        print('%d [loss: %f]' % (batch, avg_logs[0]))
        write_log(callback, train_names, avg_logs, batch)
        write_log(callback, val_names, v_loss, batch)

    if batch % 5000 == 0 and batch > 0:
        save_models(model)
