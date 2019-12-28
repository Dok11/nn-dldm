from surface_match.dataset import BatchGenerator


batch_generator = BatchGenerator()

# python -m cProfile -s time surface_match/test_get_batch.py 10001
for batch in range(100):
    x = batch_generator.get_batch()
