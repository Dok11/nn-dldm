import os

import numpy as np
from keras import Input, Model
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

    model = Flatten()(model)
    model = Dense(512, activation='linear')(model)
    model = Dropout(0.2)(model)

    input_models.append(model)

merged_layers = concatenate(input_models)

output = Dense(3, activation='linear')(merged_layers)
model = Model(inputs=inputs, outputs=output)
model.compile(optimizer=Adam(0.00001, 0.5), loss='mean_squared_error', metrics=['acc', 'mae'])
model.summary()


# ============================================================================
# --- Train and print accuracy -----------------------------------------------
# ----------------------------------------------------------------------------

file_data = get_dataset()

for epoch in range(100000):
    idx = np.random.randint(0, len(file_data['x1']), 32)
    images_x1 = file_data['x1'][idx]
    images_x2 = file_data['x2'][idx]
    train_y = file_data['y'][idx]

    model.train_on_batch(x=[images_x1, images_x2], y=train_y)

    if epoch % 250 == 0:
        m_loss = model.evaluate(x=[images_x1, images_x2], y=train_y, verbose=0)
        print('%d [D loss: %f, acc.: %.2f%%]' % (epoch, m_loss[0], 100 * m_loss[1]))

        idx_p = [0]
        images_x1_p = file_data['x1'][idx_p]
        images_x2_p = file_data['x2'][idx_p]
        train_y_p = file_data['y'][idx_p]

        predict = model.predict(x=[images_x1_p, images_x2_p])
        print('predict', predict[0])
        print('train  ', train_y_p[0])
