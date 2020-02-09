import hashlib
import json
import os

from tensorflow.python.keras import Model
from tensorflow.python.keras.preprocessing.image import save_img
from tensorflow.python.keras.utils import Progbar

from surface_match.config import SAVED_MODEL_W, CURRENT_DIR
from surface_match.dataset import BatchGenerator
from surface_match.model import get_model


def save_file(hard_indexes):
    print('Start save unique values from ' + str(len(hard_indexes)))

    examples_hash_list = set()
    unique_hard_indexes = []

    for hard_indexes_idx in range(len(hard_indexes)):
        hard_example_group = hard_indexes[hard_indexes_idx][0]
        hard_example_index = hard_indexes[hard_indexes_idx][1]

        key = str(hard_example_group) + ':' + str(hard_example_index)
        hash_key = hashlib.md5(key.encode('utf-8')).hexdigest()

        if not (hash_key in examples_hash_list):
            unique_hard_indexes.append(hard_indexes[hard_indexes_idx])
            examples_hash_list.add(hash_key)

    print('Unique values count ' + str(len(unique_hard_indexes)))

    f = open(os.path.join(CURRENT_DIR, 'hard_indexes.json'), 'w', encoding='utf-8')
    json.dump(unique_hard_indexes, f, ensure_ascii=False, indent=2)
    f.close()


batch_size = 1000
train_examples = 17_360_000  # 17_360_000
error_threshold = 0.4
save_images = False
save_images_error_threshold = 0.7

model: Model = get_model(use_model='MobileNetV2')
model.load_weights(SAVED_MODEL_W)
model.summary()

batch_generator = BatchGenerator()
batch_generator.load_dataset()
batch_generator.default_weight = 0.2 ** 2
batch_generator.train_batch_size = batch_size
batch_generator.init_weights()
batch_generator.init_weight_normalize()

hard_indexes = []
test_batches = int(2 * train_examples / batch_size)

samples = []

print('test_batches: ' + str(test_batches))
progress_bar = Progbar(test_batches)

for batch in range(test_batches):
    (t_images_1, t_images_2, t_results, indexes) = batch_generator.get_batch_train()
    result = model.predict(x=[t_images_1, t_images_2])
    samples.append((indexes, result, t_results))

    for index in range(batch_size):
        real_result = t_results[index]
        predicted_result = result[index][0]
        delta = abs(predicted_result - real_result)

        if delta > error_threshold:
            # print('Save b:' + str(batch) + '/' + str(test_batches) + ' i:' + str(index) + '/' + str(batch_size))
            hard_indexes.append([
                int(indexes[index][0]),
                int(indexes[index][1]),
                delta
            ])

        if delta > save_images_error_threshold and save_images:
            r_val = 'r%.2f' % real_result
            p_val = 'p%.2f' % predicted_result

            root_name = str(batch) + '-' + str(index) + '-root_' + p_val + 'vs' + r_val + '.jpg'
            target_name = str(batch) + '-' + str(index) + '-target_' + p_val + 'vs' + r_val + '.jpg'

            save_img('bad_predictions/' + root_name, t_images_1[index])
            save_img('bad_predictions/' + target_name, t_images_2[index])

    if batch % 5 == 0 and batch > 0:
        progress_bar.add(5)

    if batch % 20 == 0 and batch > 0:
        save_file(hard_indexes)

# Update weight complexity
batch_generator.load_example_weights()

for sample in samples:
    batch_generator.update_weights(sample[0], sample[1], sample[2])

batch_generator.save_example_weights()

save_file(hard_indexes)
