from surface_match.config import SIZE_X, SIZE_Y, GROUP_COUNT
from surface_match.dataset import get_dataset, get_batch


(train, valid, images) = get_dataset(SIZE_X, SIZE_Y)
batch_size = 30

# python -m cProfile -s time surface_match/test_get_batch.py 10001
for batch in range(100):
    x = get_batch(train, images, batch_size, GROUP_COUNT)
