import os
import shutil

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.callbacks_v1 import TensorBoard

from surface_match.config import SIZE_X, SIZE_Y, GROUP_COUNT, CURRENT_DIR, SAVED_MODEL_W
from surface_match.dataset import get_dataset, get_experimental_dataset, get_batch
from surface_match.model import get_model, save_models

np.random.seed(0)
tf.random.set_random_seed(0)


# ============================================================================
# --- Gets dataset with x1, x2 and result as y -------------------------------
# ----------------------------------------------------------------------------

# tensorboard --logdir=./logs --host=127.0.0.1
shutil.rmtree(os.path.join(CURRENT_DIR, 'logs'), ignore_errors=True)


def write_log(callback_fn, names, current_logs, batch_no):
    for name, value in zip(names, current_logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback_fn.writer.add_summary(summary, batch_no)
        callback_fn.writer.flush()


def save_models(model_for_save):
    model_for_save.save(SAVED_MODEL)
    model_for_save.save_weights(SAVED_MODEL_W)


# ============================================================================
# --- Get neural network -----------------------------------------------------
# ----------------------------------------------------------------------------

model = get_model()

if os.path.isfile(SAVED_MODEL_W):
    model.load_weights(SAVED_MODEL_W)
    print('weights are loaded')


# ============================================================================
# --- Train and print accuracy -----------------------------------------------
# ----------------------------------------------------------------------------

callback = TensorBoard('./logs')
callback.set_model(model)
train_names = ['train_loss']
val_names = ['val_loss']

(train, valid, images) = get_dataset(SIZE_X, SIZE_Y)
train_batch_size = 120

experimental_batch_t = get_experimental_dataset(True)
experimental_batch_v = get_experimental_dataset(False)

# train
start_batch = 0
sum_logs = []
for batch_index in range(50000001):
    batch = batch_index + start_batch
    (t_images_1, t_images_2, t_results) = get_batch(train, images, train_batch_size, GROUP_COUNT)

    logs = model.train_on_batch(x=[t_images_1, t_images_2], y=t_results)
    sum_logs.append(logs)

    if batch % 200 == 0 and batch > 0:
        # check model on the validation data
        (v_images_1, v_images_2, v_results) = get_batch(valid, images, train_batch_size * 3, GROUP_COUNT)
        v_loss = model.test_on_batch(x=[v_images_1, v_images_2], y=v_results)

        avg_logs = np.average(sum_logs, axis=0)
        sum_logs = []

        print('%d [loss: %f] [v. loss: %f] on batch=%d' % (batch, avg_logs[0], v_loss[0], train_batch_size))
        write_log(callback, train_names, avg_logs, batch)
        write_log(callback, val_names, v_loss, batch)

    if batch % 5000 == 0 and batch > 0:
        save_models(model)
