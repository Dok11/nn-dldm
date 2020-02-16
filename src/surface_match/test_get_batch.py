import time
import sys

sys.path.append('D:/Projects/DLDM/src')

from surface_match.dataset import BatchGenerator

batch_generator = BatchGenerator()
batch_generator.load_dataset()

# python -m cProfile -s time surface_match/test_get_batch.py
for batch in range(10):
    millis = int(round(time.time() * 1000))
    batch_generator.get_batch_train()
    end_millis = int(round(time.time() * 1000))
    print((end_millis - millis) / 1000)
