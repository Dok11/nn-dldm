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

SIZE_X = 64  # 224 is native value
SIZE_Y = 64  # 224 is native value
IMG_SHAPE = (SIZE_Y, SIZE_X, 3)
CURRENT_DIR: str = os.getcwd()
SAVED_MODEL: str = os.path.join(CURRENT_DIR, '..', '..', 'models', 'surface_match', 'model.h5')
SAVED_MODEL_W: str = os.path.join(CURRENT_DIR, '..', '..', 'models', 'surface_match', 'model_w.h5')
GROUP_COUNT = 10  # Source from src/surface_match/prepare_data.py


# ============================================================================
# --- Gets dataset with x1, x2 and result as `y` -----------------------------
# ----------------------------------------------------------------------------

def column(matrix, i):
    return np.array([row[i] for row in matrix])


def get_dataset():
    file_name = 'data_' + str(SIZE_X) + 'x' + str(SIZE_Y) + '.npz'
    file_path = os.path.join(CURRENT_DIR, '..', '..', 'train-data', 'surface_match', file_name)
    file_data = np.load(file_path, allow_pickle=True)

    return (
        file_data['train'],
        file_data['valid'],
        np.array(file_data['images']),
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

    return Model(shared_input, shared_layer, name='shared_model')


shared_model = get_image_branch()
shared_model.summary()

image_a = Input(IMG_SHAPE)
image_b = Input(IMG_SHAPE)

branch_a = shared_model(image_a)
branch_b = shared_model(image_b)

merged_layers = concatenate([branch_a, branch_b])

merged_layers = Flatten()(merged_layers)
merged_layers = Dense(1024, activation='selu')(merged_layers)
merged_layers = Dense(1024, activation='selu')(merged_layers)

output = Dense(1, kernel_initializer='normal', activation='selu')(merged_layers)
model = Model(inputs=[image_a, image_b], outputs=output)
model.compile(optimizer=tf.keras.optimizers.Adam(0.00005, decay=0.00001),
              loss='mse',
              metrics=[tf.keras.metrics.Accuracy()])

model.summary()

# if os.path.isfile(SAVED_MODEL_W):
#     model.load_weights(SAVED_MODEL_W)
#     print('weights are loaded')


# ============================================================================
# --- Train and print accuracy -----------------------------------------------
# ----------------------------------------------------------------------------

callback = TensorBoard('./logs')
callback.set_model(model)
train_names = ['train_loss']
val_names = ['val_loss']

(train, valid, images) = get_dataset()
train_batch_size = 60


def get_batch(data_groups):
    samples_per_group = train_batch_size // GROUP_COUNT

    images_1 = []
    images_2 = []
    results = []

    for group_index in range(GROUP_COUNT):

        group = np.array(data_groups[group_index])
        group_samples_indexes = np.random.randint(0, len(group), samples_per_group)
        group_samples = group[group_samples_indexes]

        group_images_1_idx = column(group_samples, 0)
        group_images_2_idx = column(group_samples, 1)
        group_images_1 = images[group_images_1_idx.astype(int)]
        group_images_2 = images[group_images_2_idx.astype(int)]
        group_results = column(group_samples, 2)

        for sample_index in range(len(group_results)):
            images_1.append(group_images_1[sample_index] / 255.)
            images_2.append(group_images_2[sample_index] / 255.)
            results.append(group_results[sample_index])

    return images_1, images_2, results


# train
sum_logs = []
for batch in range(50000001):
    (t_images_1, t_images_2, t_results) = get_batch(train)

    logs = model.train_on_batch(x=[t_images_1, t_images_2], y=t_results)
    sum_logs.append(logs)

    if batch % 200 == 0 and batch > 0:
        # check model on the validation data
        (v_images_1, v_images_2, v_results) = get_batch(valid)
        v_loss = model.test_on_batch(x=[v_images_1, v_images_2], y=v_results)

        avg_logs = np.average(sum_logs, axis=0)
        sum_logs = []

        print('%d [loss: %f]' % (batch, avg_logs[0]))
        write_log(callback, train_names, avg_logs, batch)
        write_log(callback, val_names, v_loss, batch)

    if batch % 5000 == 0 and batch > 0:
        save_models(model)