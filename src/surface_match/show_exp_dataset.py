from tensorflow.python.keras.preprocessing.image import save_img

from surface_match.dataset import BatchGenerator, get_experimental_dataset


batch_generator = BatchGenerator()
# batch_generator.load_dataset()
batch_generator.default_weight = 0.06 ** 2
batch_generator.init_weights()
batch_generator.load_example_weights()
batch_generator.init_weight_normalize()

(t_images_1, t_images_2, t_results, indexes) = get_experimental_dataset(True)

for index in range(len(t_results)):
    val = str(t_results[index])

    root_name = str(index) + '-root_' + val + '.jpg'
    target_name = str(index) + '-target_' + val + '.jpg'

    save_img('exp_dataset/' + root_name, t_images_1[index])
    save_img('exp_dataset/' + target_name, t_images_2[index])
