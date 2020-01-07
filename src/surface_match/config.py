import os


FILE_NAME_TRAIN = 'experiment_train'
FILE_NAME_VALID = 'experiment_valid'

SIZE_X = 224  # 224 is native value
SIZE_Y = 224  # 224 is native value
CHANNELS = 3

IMG_SHAPE = (SIZE_Y, SIZE_X, CHANNELS)
GROUP_COUNT = 10

CURRENT_DIR: str = os.getcwd()
SAVED_MODEL: str = os.path.join(CURRENT_DIR, '..', '..', 'models', 'surface_match', 'model.h5')
SAVED_MODEL_W: str = os.path.join(CURRENT_DIR, '..', '..', 'models', 'surface_match', 'model_w.h5')
