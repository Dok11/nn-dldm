import os
import shutil
import time
import gc
from functools import reduce

import torch
import torch.nn.functional as f
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from surface_match.config import SAVED_MODEL_W, CURRENT_DIR
from surface_match.surface_match_dataset import SurfaceMatchDataset


def memory_stats(device):
    unit = 'MB'
    unit_denom = float(1024 * 1024)

    allocated = torch.cuda.memory_allocated(device) / unit_denom
    cached = torch.cuda.memory_cached(device) / unit_denom

    return f'<{device}> - allocated/cached: {allocated:.1f} {unit}/{cached:.1f} {unit}'


def gc_tensors():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size())
        except:
            pass


# ============================================================================
# --- Get neural network -----------------------------------------------------
# ----------------------------------------------------------------------------

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_branch = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # nn.Conv2d(38, 42, 3, padding=1),
            # nn.BatchNorm2d(42),
            # nn.ReLU(),
            # nn.MaxPool2d(2),
            #
            # nn.Conv2d(42, 48, 3, padding=1),
            # nn.BatchNorm2d(48),
            # nn.ReLU(),
            # nn.MaxPool2d(2),
            #
            # nn.Conv2d(48, 64, 3, padding=1),
            # nn.BatchNorm2d(64),
            # nn.ReLU(),
            # nn.MaxPool2d(2),
        )

        self.avgpooling = nn.AdaptiveMaxPool2d((2, 2))

        self.fc1 = nn.Linear(64 * 2 * 2 * 2, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        print('requires_grad:', x1.requires_grad, x2.requires_grad)

        # Image branches
        x1 = self.conv_branch(x1)
        x1 = self.avgpooling(x1)

        x2 = self.conv_branch(x2)
        x2 = self.avgpooling(x2)

        # Flatten
        x1 = x1.view(-1, self.num_flat_features(x1))
        x2 = x2.view(-1, self.num_flat_features(x2))

        x = torch.cat((x1, x2), dim=1)

        x = f.relu(self.fc1(x))
        x = self.fc2(x).view(-1)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def params_count(self):
        """count number trainable parameters in a pytorch model"""
        return sum(reduce(lambda a, b: a * b, x.size()) for x in self.parameters())


model = Model().to(device)
# model.load_state_dict(torch.load(SAVED_MODEL_W))

print(model)
print('Params count: %d' % model.params_count())

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


# ============================================================================
# --- Create logger ----------------------------------------------------------
# ----------------------------------------------------------------------------

# tensorboard --logdir=./logs --host=127.0.0.1
shutil.rmtree(os.path.join(CURRENT_DIR, 'logs'), ignore_errors=True)
writer = SummaryWriter(log_dir='./logs')


# ============================================================================
# --- Data loader ------------------------------------------------------------
# ----------------------------------------------------------------------------

transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                transforms.RandomVerticalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5], [0.5])])

dataset = SurfaceMatchDataset(transform=transform)
dataset.load_dataset()

data_loader = DataLoader(dataset, batch_size=32, num_workers=0)


# ============================================================================
# --- Train ------------------------------------------------------------------
# ----------------------------------------------------------------------------


train_on_batch_time = 0.0
log_every_batches = 100
train_loss = 0.0
accuracy = 0.0

for i_batch in range(100_000_000):
    millis = time.time()

    dataset.use_valid = False
    sample = next(iter(data_loader))

    input_images_1: torch.Tensor = sample['image_1'].to(device)
    input_images_2: torch.Tensor = sample['image_2'].to(device)
    real_outputs = sample['result'].to(device)

    outputs: torch.Tensor = model(input_images_1, input_images_2)

    loss = criterion(outputs, real_outputs)
    train_loss += float(loss)
    accuracy += float(torch.sum(torch.abs(outputs - real_outputs))) / len(real_outputs)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_on_batch_time += time.time() - millis

    if i_batch % log_every_batches == 0 and i_batch > 0:
        train_loss_avg = train_loss / log_every_batches

        dataset.use_valid = True
        val_sample = next(iter(data_loader))

        val_input_images_1 = sample['image_1'].to(device)
        val_input_images_2 = sample['image_2'].to(device)
        val_real_outputs = sample['result'].to(device)

        val_outputs: torch.Tensor = model(val_input_images_1, val_input_images_2)
        val_loss = criterion(val_outputs, val_real_outputs)
        val_accuracy = torch.sum(torch.abs(val_outputs - val_real_outputs)).item() / len(val_real_outputs)

        batch_num = i_batch + 0

        print('%d [loss: %f] [v. loss: %f] [train time: %f]'
              % (batch_num, train_loss_avg, val_loss.item(), train_on_batch_time))

        writer.add_scalars('Loss', {
            'Train': train_loss_avg,
            'Valid': val_loss.item(),
        }, batch_num)

        writer.add_scalars('Accuracy', {
            'Train': accuracy / log_every_batches,
            'Valid': val_accuracy,
        }, batch_num)

        train_on_batch_time = 0.0
        train_loss = 0.0
        accuracy = 0.0

    if i_batch % 1000 == 0 and i_batch > 0:
        torch.save(model.state_dict(), SAVED_MODEL_W)
        print('Model saved')
