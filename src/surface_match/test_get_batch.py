import time

from surface_match.dataset import BatchGenerator

batch_generator = BatchGenerator()
batch_generator.load_dataset()
batch_generator.init_weights()
batch_generator.load_example_weights()
batch_generator.init_weight_normalize()

millis = int(round(time.time() * 1000))
# python -m cProfile -s time surface_match/test_get_batch.py
for batch in range(100):
    x = batch_generator.get_batch_train()

end_millis = int(round(time.time() * 1000))
print((end_millis - millis) / 1000)
