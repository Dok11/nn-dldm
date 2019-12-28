import hashlib
import json
import os

from tensorflow.python.keras import Model
from tensorflow.python.keras.preprocessing.image import save_img

from surface_match.config import SAVED_MODEL_W, CURRENT_DIR
from surface_match.dataset import BatchGenerator
from surface_match.model import get_model

batch_size = 1000
train_examples = 1536000

model: Model = get_model()
model.load_weights(SAVED_MODEL_W)
model.summary()
save_images = False

batch_generator = BatchGenerator()
batch_generator.train_batch_size = batch_size

hard_indexes = []
test_batches = int(10 * train_examples / batch_generator.train_batch_size)

for batch in range(test_batches):
    (t_images_1, t_images_2, t_results, indexes) = batch_generator.get_batch()
    result = model.predict(x=[t_images_1, t_images_2])

    for index in range(batch_size):
        real_result = t_results[index]
        predicted_result = result[index][0]
        delta = abs(predicted_result - real_result)

        if delta > 0.33:
            print('Save b:' + str(batch) + '/' + str(test_batches) + ' i:' + str(index) + '/' + str(batch_size))
            hard_indexes.append([
                int(indexes[index][0]),
                int(indexes[index][1]),
                delta
            ])

            if save_images:
                r_val = 'r%.2f' % real_result
                p_val = 'p%.2f' % predicted_result

                root_name = str(batch) + '-' + str(index) + '-root_' + p_val + 'vs' + r_val + '.jpg'
                target_name = str(batch) + '-' + str(index) + '-target_' + p_val + 'vs' + r_val + '.jpg'

                save_img('bad_predictions/' + root_name, t_images_1[index])
                save_img('bad_predictions/' + target_name, t_images_2[index])


print('Start save unique values')

examples_hash_list = []
unique_hard_indexes = []

for i in range(len(hard_indexes)):
    hard_example_group = hard_indexes[i][0]
    hard_example_index = hard_indexes[i][1]

    key = str(hard_example_group) + ':' + str(hard_example_index)
    hash_key = hashlib.md5(key.encode('utf-8')).hexdigest()

    if not (hash_key in examples_hash_list):
        unique_hard_indexes.append(hard_indexes[i])
        examples_hash_list.append(hash_key)


f = open(os.path.join(CURRENT_DIR, 'hard_indexes.json'), 'w', encoding='utf-8')
json.dump(unique_hard_indexes, f, ensure_ascii=False, indent=2)
f.close()
