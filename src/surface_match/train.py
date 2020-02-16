import os
import shutil
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.callbacks_v1 import TensorBoard

from surface_match.config import CURRENT_DIR, SAVED_MODEL_W
from surface_match.dataset import BatchGenerator, get_experimental_dataset
from surface_match.model import get_model, save_models

# seed = 1
# np.random.seed(seed)
# tf.compat.v1.random.set_random_seed(seed)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


# ============================================================================
# --- Get neural network -----------------------------------------------------
# ----------------------------------------------------------------------------

model = get_model(use_model='MobileNetV2')
model.summary()

if os.path.isfile(SAVED_MODEL_W):
    model.load_weights(SAVED_MODEL_W)
    print('weights are loaded')
    pass


# ============================================================================
# --- Create logger ----------------------------------------------------------
# ----------------------------------------------------------------------------

# tensorboard --logdir=./logs --host=127.0.0.1
shutil.rmtree(os.path.join(CURRENT_DIR, 'logs'), ignore_errors=True)

callback = TensorBoard('./logs')
callback.set_model(model)


def write_log(callback_fn, names, current_logs, batch_no):
    for name, value in zip(names, current_logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback_fn.writer.add_summary(summary, batch_no)
        callback_fn.writer.flush()


# ============================================================================
# --- Train and print accuracy -----------------------------------------------
# ----------------------------------------------------------------------------

batch_generator = BatchGenerator()
batch_generator.train_batch_size = 120
batch_generator.load_dataset()
# batch_generator.default_weight = 0.5 ** 2
# batch_generator.init_weights()
# batch_generator.load_example_weights()
# batch_generator.init_weight_normalize()

# batch_exp_train = get_experimental_dataset(True)
# batch_exp_valid = get_experimental_dataset(False)

# train
start_batch = 0
sum_logs = []
train_on_batch_time = 0

for batch_index in range(50000001):
    batch = batch_index + start_batch

    (t_images_1, t_images_2, t_results, indexes) = batch_generator.get_batch_train()

    millis = time.time()
    logs = model.train_on_batch(x=[t_images_1, t_images_2], y=t_results)
    train_on_batch_time += time.time() - millis

    sum_logs.append(logs)

    if batch % 50 == 0 and batch > 0:
        # Check model on the validation data
        (v_images_1, v_images_2, v_results, indexes) = batch_generator.get_batch_valid()
        v_loss = model.test_on_batch(x=[v_images_1, v_images_2], y=v_results)

        avg_logs = np.average(sum_logs, axis=0)
        sum_logs = []

        print('%d [loss: %f] [v. loss: %f] [train time: %f]' % (batch, avg_logs[1], v_loss[1], train_on_batch_time))
        write_log(callback, ['train_loss', 't_loss_in_fact'], avg_logs, batch)
        write_log(callback, ['val_loss', 'v_loss_in_fact'], v_loss, batch)
        train_on_batch_time = 0
        summary_time = 0

    if batch % 2000 == 0 and batch > 0:
        # Update weights complexity
        # batch_generator.update_weights_by_model(model)
        pass

    if batch % 1000 == 0 and batch > 0:
        save_models(model)
        # batch_generator.save_example_weights()
