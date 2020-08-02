import torch
import torch.nn.functional as f
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from surface_match.config import SIZE_Y, SIZE_X
from surface_match.surface_match_dataset import SurfaceMatchDataset

# ============================================================================
# --- Train and print accuracy -----------------------------------------------
# ----------------------------------------------------------------------------

# ============================================================================
# --- Get neural network -----------------------------------------------------
# ----------------------------------------------------------------------------

# ============================================================================
# --- Create logger ----------------------------------------------------------
# ----------------------------------------------------------------------------

# tensorboard --logdir=./logs --host=127.0.0.1
# shutil.rmtree(os.path.join(CURRENT_DIR, 'logs'), ignore_errors=True)
#
# callback = TensorBoard('./logs')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d( 3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)

        self.fc1 = nn.Linear(14*14*128, 1024)
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(1024, 1)

    def forward(self, x1: torch.Tensor):
        x1 = f.relu(self.conv1(x1))
        x1 = f.max_pool2d(x1, 2, 2)

        x1 = f.relu(self.conv2(x1))
        x1 = f.max_pool2d(x1, 2, 2)

        x1 = f.relu(self.conv3(x1))
        x1 = f.max_pool2d(x1, 2, 2)

        x1 = f.relu(self.conv4(x1))
        x1 = f.max_pool2d(x1, 2, 2)

        x1 = x1.view(-1, self.num_flat_features(x1))

        x1 = f.relu(self.fc1(x1))
        x1 = self.dropout1(x1)

        x1 = self.fc2(x1).view(-1)

        return x1

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


model = Model().to(device)

print(model)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)


# file_path = os.path.join(os.getcwd(), '..', '..', 'train-data', 'surface_match', FILE_NAME_DATA + str('.npz'))
# exp_sample = np.load(file_path, allow_pickle=True)['sample']
# model.forward(exp_sample)

transform_train = transforms.Compose([transforms.Resize((SIZE_Y, SIZE_X)),
                                      # transforms.RandomHorizontalFlip(),
                                      # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.5], [0.5])])

dataset = SurfaceMatchDataset(transform=transform_train)
dataset.load_dataset()
dataset.init_weights()
dataset.init_weight_normalize()

data_loader = DataLoader(dataset, batch_size=4, num_workers=0)

for i_batch, sample_batched in enumerate(data_loader):
    input_images_1 = sample_batched['image_1'].to(device)
    input_images_2 = sample_batched['image_2'].to(device)
    real_outputs = sample_batched['result'].to(device)

    outputs: torch.Tensor = model(input_images_1)
    loss = criterion(outputs, real_outputs)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(loss.item())
