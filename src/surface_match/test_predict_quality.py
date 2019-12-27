import json
import os

from tensorflow.python.keras import Model
from tensorflow.python.keras.preprocessing.image import save_img

from surface_match.config import SIZE_X, SIZE_Y, SAVED_MODEL_W, GROUP_COUNT, CURRENT_DIR
from surface_match.dataset import get_dataset, get_batch
from surface_match.model import get_model

(train, valid, images) = get_dataset(SIZE_X, SIZE_Y)

train_batch_size = 30

model: Model = get_model()
model.load_weights(SAVED_MODEL_W)
model.summary()

hard_indexes = []
test_batches = int(3 * 1000 * 1000 / train_batch_size)

for batch in range(test_batches):
    (t_images_1, t_images_2, t_results, indexes) = get_batch(train, images, train_batch_size, GROUP_COUNT)
    result = model.predict(x=[t_images_1, t_images_2])

    for index in range(train_batch_size):
        real_result = t_results[index]
        predicted_result = result[index][0]
        delta = abs(predicted_result - real_result)

        if delta > 0.4:
            r_val = 'r%.2f' % real_result
            p_val = 'p%.2f' % predicted_result

            root_name = str(batch) + '-' + str(index) + '-root_' + p_val + 'vs' + r_val + '.jpg'
            target_name = str(batch) + '-' + str(index) + '-target_' + p_val + 'vs' + r_val + '.jpg'

            # save_img('bad_predictions/' + root_name, t_images_1[index])
            # save_img('bad_predictions/' + target_name, t_images_2[index])

            print('Save b:' + str(batch) + ' i:' + str(index))
            hard_indexes.append([
                int(indexes[index][0]),
                int(indexes[index][1]),
                delta
            ])

f = open(os.path.join(CURRENT_DIR, 'hard_indexes.json'), 'w', encoding='utf-8')
json.dump(hard_indexes, f, ensure_ascii=False, indent=2)
f.close()
