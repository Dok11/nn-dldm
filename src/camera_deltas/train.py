import os

import tensorflow as tf
import numpy as np
from keras import Input, Model
from keras.callbacks import TensorBoard
from keras.layers import Conv2D, LeakyReLU, Dropout, BatchNormalization, Flatten, Dense, concatenate
from keras.optimizers import Adam

# ============================================================================
# --- GLOBAL PARAMS ----------------------------------------------------------
# ----------------------------------------------------------------------------

SIZE_X = 90
SIZE_Y = 60
INPUT_NUMS = 2
IMG_SHAPE = (SIZE_Y, SIZE_X, 1)
CURRENT_DIR: str = os.getcwd()
SAVED_MODEL: str = os.path.join(CURRENT_DIR, '..', '..', 'models', 'camera-pose', 'model.h5')
SAVED_MODEL_W: str = os.path.join(CURRENT_DIR, '..', '..', 'models', 'camera-pose', 'model_w.h5')


# ============================================================================
# --- Gets dataset with x1, x2 and result as `y` -----------------------------
# ----------------------------------------------------------------------------

def get_dataset():
    file_path = os.path.join(CURRENT_DIR, '..', '..', 'train-data', 'deltas', 'data_000.npz')
    file_data = np.load(file_path)

    result = {
        'x1': file_data['x1'] / 255.,
        'x2': file_data['x2'] / 255.,
        'y': file_data['y'],
    }

    return result


# tensorboard --logdir=./logs --host=127.0.0.1
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

inputs = []
input_models = []

for input_idx in range(INPUT_NUMS):
    model_input = Input(shape=IMG_SHAPE)
    inputs.append(model_input)

    # 90x60 -> 45x30
    model = Conv2D(64, kernel_size=3, strides=2, input_shape=IMG_SHAPE, padding='same')(model_input)
    model = LeakyReLU(alpha=0.2)(model)
    model = Dropout(0.2)(model)

    # 45x30 -> 23x15
    model = Conv2D(128, kernel_size=3, strides=2, padding='same')(model)
    model = BatchNormalization(momentum=0.8)(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = Dropout(0.2)(model)

    # 23x15 -> 12x8
    model = Conv2D(256, kernel_size=3, strides=2, padding='same')(model)
    model = BatchNormalization(momentum=0.8)(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = Dropout(0.2)(model)

    input_models.append(model)

merged_layers = concatenate(input_models)

merged_layers = Flatten()(merged_layers)
merged_layers = Dense(512, activation='linear')(merged_layers)
merged_layers = Dropout(0.2)(merged_layers)

output = Dense(3, activation='linear')(merged_layers)
model = Model(inputs=inputs, outputs=output)
model.compile(optimizer=Adam(0.00001, 0.5), loss='mean_squared_error', metrics=['mae'])
model.summary()


# ============================================================================
# --- Train and print accuracy -----------------------------------------------
# ----------------------------------------------------------------------------

log_path = './logs'
callback = TensorBoard(log_path)
callback.set_model(model)
train_names = ['train_loss', 'train_mae']
val_names = ['val_loss', 'val_mae']

file_data = get_dataset()

# train
train_x1 = file_data['x1'][:8000]
train_x2 = file_data['x2'][:8000]
train_y = file_data['y'][:8000]

# test
test_x1 = file_data['x1'][8000:]
test_x2 = file_data['x2'][8000:]
test_y = file_data['y'][8000:]

# predict
idx_p = [0]
images_x1_p = file_data['x1'][idx_p]
images_x2_p = file_data['x2'][idx_p]
train_y_p = file_data['y'][idx_p]

# train
for batch in range(100000):
    idx = np.random.randint(0, len(train_x1), 64)
    images_x1 = train_x1[idx]
    images_x2 = train_x2[idx]
    images_y = train_y[idx]

    logs = model.train_on_batch(x=[images_x1, images_x2], y=images_y)

    if batch % 500 == 0:
        test_idx = np.random.randint(0, len(test_x1), 64)
        m_loss = model.test_on_batch(x=[test_x1[test_idx], test_x2[test_idx]], y=test_y[test_idx])
        print('%d [D loss: %f, acc.: %.2f%%]' % (batch, m_loss[0], 100 * m_loss[1]))

        predict = model.predict(x=[test_x1, test_x2])
        print('predict', predict[0])
        print('train  ', train_y_p[0])
        write_log(callback, train_names, logs, batch)
        write_log(callback, val_names, m_loss, batch)

        # save_models(model)
