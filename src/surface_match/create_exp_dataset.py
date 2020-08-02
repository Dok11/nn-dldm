import os

import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms

from surface_match.config import SIZE_Y, SIZE_X, FILE_NAME_DATA
from surface_match.surface_match_dataset import SurfaceMatchDataset


def save(file_name: str, sample: list):
    train_dir = os.path.join(os.getcwd(), '..', '..', 'train-data', 'surface_match')

    file = os.path.join(train_dir, file_name)
    np.savez(file, sample=sample)


transform_train = transforms.Compose([transforms.Resize((SIZE_Y, SIZE_X)),
                                      # transforms.RandomHorizontalFlip(),
                                      # transforms.RandomRotation(10),
                                      # transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
                                      # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.5], [0.5])])

dataset = SurfaceMatchDataset(transform=transform_train)
dataset.load_dataset()
dataset.init_weights()
dataset.init_weight_normalize()

data_loader = DataLoader(dataset, batch_size=128, num_workers=0)
i_batch = 0
sample_batched = None

for i_batch, sample_batched in enumerate(data_loader):
    break

save(FILE_NAME_DATA, sample=sample_batched)
