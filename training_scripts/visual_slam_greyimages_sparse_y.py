
# Libraries
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Pytorch
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR
#from torchviz import make_dot # for visualizing the architecture as a graph

# FrEIA imports
import FrEIA.framework as Ff
import FrEIA.modules as Fm
from FrEIA.framework import InputNode, OutputNode, Node, ReversibleGraphNet, ConditionNode
from FrEIA.modules import GLOWCouplingBlock, PermuteRandom, RNVPCouplingBlock

# cuda
from torch.cuda.amp import GradScaler, autocast
import time

import os
import re
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

device='cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)
print(device)

#partially_trained_model_path ="proj_cameralocal_INN/local_INN_model_best.pt"
# Folder containing the images
folder_path =  "large" #"images_hall_dense"

ENCODING_LENGTH =20
print("ENCODING LENGTH=",ENCODING_LENGTH)
GLOW_COUPLINGBLOCK_FACTOR =1
print("GLOW_COUPLINGBLOCK_FACTOR =", GLOW_COUPLINGBLOCK_FACTOR )

LATENT_DIM = 6*ENCODING_LENGTH -6  #96
POSE_DIM = 6
IMAGE_CHANNELS = 1  # Assuming RGB images

BATCHSIZE = int(500)
print("batch size",BATCHSIZE )

N_DIM = int(LATENT_DIM+6)
COND_DIM = 6
COND_OUT_DIM = 12
LR = 5e-4

INSTABILITY_RECOVER = 1
USE_MIX_PRECISION_TRAINING = 0

seed_value = 42
np.random.seed(seed_value)


# for generating conditional pose
mean = 0

std_dev = 0.5
lower_bound = -2
upper_bound = 2

std_dev_theta = 11.5
theta_lower_bound = -57.3
theta_upper_bound = 57.3

def generate_valid_value(mean, std_dev, lower_bound, upper_bound):
    while True:
        value = np.random.normal(mean, std_dev)
        if lower_bound <= value <= upper_bound:
            return value


# ---- Custom Dataset ----
class HouseImageDataset_bk(Dataset):
    def __init__(self, image_paths, poses, cond_poses, transform=None):
        self.image_paths = image_paths  # List of image file paths
        self.poses = poses              # Corresponding (x, y, yaw) for each image
        self.cond_poses = cond_poses
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image = plt.imread(self.image_paths[idx])  # Assuming images are grayscale or RGB
        if self.transform:
            image = self.transform(image)

        # Load pose
        pose = torch.tensor(self.poses[idx], dtype=torch.float32)
        cond_pose = torch.tensor(self.cond_poses[idx], dtype=torch.float32)
        return image.to(device), pose.to(device), cond_pose.to(device)


class HouseImageDataset_bk2(Dataset):
    def __init__(self, image_paths, poses, cond_poses, transform=None):
        self.device = device 
        self.transform = transform
        print("device val in HouseImageDataset is", self.device)
        start_time = time.time()
        print("Loading images to GPU...")

        # Preload and transform images
        image_tensors = []
        for i, path in enumerate(image_paths):
            image = plt.imread(path)  # Assuming images are grayscale or RGB
            if self.transform:
                image = self.transform(image)
            # already a tensor . image = torch.tensor(image, dtype=torch.float32)  # Convert to tensor
            image_tensors.append(image)
            if (i + 1) % 5000 == 0 or (i + 1) == len(image_paths):
                elapsed = time.time() - start_time
                print(f"Processed {i + 1}/{len(image_paths)} images in {elapsed:.2f} seconds")
                

        # Stack all images into a single tensor and move to GPU
        self.images = torch.stack(image_tensors).to(self.device)

        print("Loading poses and conditional poses to GPU...")
        self.poses = torch.tensor(poses, dtype=torch.float32, device=self.device) #torch.tensor(poses, dtype=torch.float32).to(self.device)
        self.cond_poses = torch.tensor(cond_poses, dtype=torch.float32, device=self.device) #torch.tensor(cond_poses, dtype=torch.float32).to(self.device)
        print(f"All data loaded to GPU in {time.time() - start_time:.2f} seconds")

    def __len__(self):
        return self.images.size(0)

    def __getitem__(self, idx):
        # Return preloaded data from GPU memory
        return self.images[idx], self.poses[idx], self.cond_poses[idx]
    
class HouseImageDataset_bk3(Dataset):
    def __init__(self, image_paths, poses, cond_poses, transform=None):
        self.device = device 
        self.transform = transform
        print("device val in HouseImageDataset is", self.device)
        start_time = time.time()
        print("Loading images to GPU...")

        # Preload and transform images
        num_images = len(poses)
        print("num images is", num_images)
        print("trying to allocate tensor on cuda")
        pre_alloc_tensor = torch.empty((num_images,1, 256,256), device=device)
        print("tensor successfully preallocated")
        for i, path in enumerate(image_paths):
            image = plt.imread(path)  # Assuming images are grayscale or RGB
            if self.transform:
                image = self.transform(image)
            # already a tensor . image = torch.tensor(image, dtype=torch.float32)  # Convert to tensor
            pre_alloc_tensor[i]= image
            if (i + 1) % 5000 == 0 or (i + 1) == len(image_paths):
                elapsed = time.time() - start_time
                print(f"Processed {i + 1}/{len(image_paths)} images in {elapsed:.2f} seconds")
                

        # Stack all images into a single tensor and move to GPU
        self.images = pre_alloc_tensor #torch.stack(image_tensors).to(self.device)

        print("Loading poses and conditional poses to GPU...")
        self.poses = torch.tensor(poses, dtype=torch.float32, device=self.device) #torch.tensor(poses, dtype=torch.float32).to(self.device)
        self.cond_poses = torch.tensor(cond_poses, dtype=torch.float32, device=self.device) #torch.tensor(cond_poses, dtype=torch.float32).to(self.device)
        print(f"All data loaded to GPU in {time.time() - start_time:.2f} seconds")

    def __len__(self):
        return self.images.size(0)

    def __getitem__(self, idx):
        # Return preloaded data from GPU memory
        return self.images[idx], self.poses[idx], self.cond_poses[idx]
    
class HouseImageDataset(Dataset):
    def __init__(self, image_paths, poses, cond_poses, transform=None):
        self.image_paths = image_paths  # List of image file paths
        self.poses = poses              # Corresponding (x, y, yaw) for each image
        self.cond_poses = cond_poses
        self.transform = transform
        
        num_images = len(poses)
        self.image_tensors_array = torch.empty(num_images, 1, 256, 256)
        for idx in range(num_images):
            image = plt.imread(self.image_paths[idx])  # Assuming images are grayscale or RGB
            if self.transform:
                image = self.transform(image)
            #print("shape of transformed image is", image.shape)
            self.image_tensors_array[idx] =image  
            if (idx + 1) % 2000 == 0 or (idx + 1) == len(image_paths):
                print(f"Processed", idx+1)
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        pose = torch.tensor(self.poses[idx], dtype=torch.float32)
        cond_pose = torch.tensor(self.cond_poses[idx], dtype=torch.float32)
        return self.image_tensors_array[idx], pose, cond_pose

# Min-Max Normalization function
def min_max_normalize_2(data, min_value, max_value):
    return (data - min_value) / (max_value - min_value)


# Lists to store file names and extracted values
file_names = []
pose_values = []  # To store x, y, yaw values as lists
cond_pose_values =[]
# Regular expression to extract x, y, yaw from the filename
pattern = re.compile(r'(-?\d*\.\d+)_(-?\d*\.\d+)_(-?\d+\.\d{2})')   #re.compile(r'(-?\d*\.?\d+)_(-?\d*\.?\d+)_(-?\d+)')
# Loop through each file in the input training images folder
for file in os.listdir(folder_path):
    if file.endswith(".jpg"):  # Process only .jpg files

        match = pattern.search(file)  # Match the file name pattern
        if match:
            x = float(match.group(1))  # Extract x value
            y = float(match.group(2))  # Extract y value
            yaw = float(match.group(3))  # Extract yaw value

            x_cond = generate_valid_value(mean, std_dev, lower_bound, upper_bound)+ x
            y_cond = generate_valid_value(mean, std_dev, lower_bound, upper_bound)+ y
            yaw_cond = generate_valid_value(mean, std_dev_theta, theta_lower_bound, theta_upper_bound)+yaw
            if(yaw_cond <0):
                yaw_cond += 360
            elif (yaw_cond >360):
                yaw_cond -= 360

            full_path = os.path.join(folder_path, file)
            file_names.append(full_path)  # Store the file name
            pose_values.append([x, y, yaw])  # Store the extracted values as a list
            cond_pose_values.append([x_cond, y_cond, yaw_cond])



## Find the min and max for x, y, and yaw
x_values = [pose[0] for pose in pose_values]
y_values = [pose[1] for pose in pose_values]
yaw_values = [pose[2] for pose in pose_values]

# # Min and max for x, y, yaw
# x_min, x_max = min(x_values), max(x_values)
# y_min, y_max = min(y_values), max(y_values)


x_min, x_max = -8.1, 8.7
y_min, y_max = -5.0, 2.6
print("values", x_min, x_max,y_min, y_max)

count =0
for id in range(len(cond_pose_values)):
    if(cond_pose_values[id][0]< x_min):
        cond_pose_values[id][0] = x_min
        count+=1
    elif (cond_pose_values[id][0] >x_max):
        cond_pose_values[id][0] = x_max
        count+=1
    if(cond_pose_values[id][1]< y_min):
        cond_pose_values[id][1]= y_min
        count+=1
    elif (cond_pose_values[id][1] >y_max):
        cond_pose_values[id][1]= y_max
        count+=1
print("number of condition poses modified =", count)



# Example: Display the first few results
print("File Names:", file_names[:5])
print("Pose Values (x, y, yaw):", pose_values[:5])

#Normalize pose data########

# Perform normalization for each
normalized_pose_values = []
normalized_cond_pose_values = []

x_cond_values = [pose[0] for pose in cond_pose_values]
y_cond_values = [pose[1] for pose in cond_pose_values]
yaw_cond_values = [pose[2] for pose in cond_pose_values]

for x, y, yaw, x_cond, y_cond, yaw_cond in zip(x_values, y_values, yaw_values, x_cond_values, y_cond_values, yaw_cond_values):
    normalized_x = min_max_normalize_2(x, x_min, x_max)
    normalized_y = min_max_normalize_2(y, y_min, y_max)
    normalized_yaw = yaw /360.0

    normalized_cond_x = min_max_normalize_2(x_cond, x_min, x_max)
    normalized_cond_y = min_max_normalize_2(y_cond, y_min, y_max)
    normalized_cond_yaw = yaw_cond /360.0

    normalized_pose_values.append([normalized_x, normalized_y, normalized_yaw])
    normalized_cond_pose_values .append([normalized_cond_x, normalized_cond_y, normalized_cond_yaw])


#put condition poses into bins
cond_pose_np = np.array(normalized_cond_pose_values)
# Multiply by 10, take the floor, then divide by 10
cond_pose_np = np.floor(cond_pose_np * 10.0) / 10.0
normalized_cond_pose_values = cond_pose_np.tolist()

# Example: Display the first few normalized values
print("Normalized Pose Values (x, y, yaw):", normalized_pose_values[:5])

print("Normalized conditioned Pose Values (x, y, yaw):", normalized_cond_pose_values[:5])

print("total length of dataset is", len(normalized_pose_values))



image_paths = file_names  # Add your image paths here
poses = normalized_pose_values #encoded_pose  # Replace with actual (x, y, yaw, and additional pose data)
cond_poses = normalized_cond_pose_values

################Create Data set
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),  # Images are now 256x256
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Split the data image_paths, poses
X_train, X_test = train_test_split(
    list(zip(image_paths, poses, cond_poses)),  # Combine the two lists into a tuple
    test_size=0.10,  # 20% for testing, adjust as needed
    random_state=42,  # Optional, for reproducibility
)

# Unzip the train and test sets back into separate lists
train_image_paths, train_poses, train_cond_poses = zip(*X_train)
test_image_paths, test_poses , test_cond_poses= zip(*X_test)

# Convert back to lists 
train_image_paths = list(train_image_paths)
train_poses = list(train_poses)
train_cond_poses = list(train_cond_poses)

test_image_paths = list(test_image_paths)
test_poses = list(test_poses)
test_cond_poses = list(test_cond_poses)

print("batch size2",BATCHSIZE )
train_dataset = HouseImageDataset(train_image_paths, train_poses, train_cond_poses, transform)
train_dataloader = DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=True, pin_memory = True )

test_dataset = HouseImageDataset(test_image_paths, test_poses,test_cond_poses, transform)
test_dataloader = DataLoader(test_dataset, batch_size=BATCHSIZE, shuffle=True, pin_memory = True)

class VariationalEncoder(nn.Module):
    def __init__(self, image_channels, latent_dim):
        super(VariationalEncoder, self).__init__()

        # Encoder: Image -> Latent Space
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2, padding=1),  # (B, 32, 128, 128)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # (B, 64, 64, 64)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # (B, 128, 32, 32)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # (B, 256, 16, 16)
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(256 * 16 * 16, latent_dim)  # Adjust size based on resolution
        self.fc_logvar = nn.Linear(256 * 16 * 16, latent_dim)
        self.N = torch.distributions.Normal(0,1)
        self.N.loc = self.N.loc.to(device) 
        self.N.scale = self.N.scale.to(device)
        self.kl = 0

    def forward(self, x):
        x_encoded = self.encoder(x)
        mu = self.fc_mu(x_encoded)
        sigma = torch.exp(self.fc_logvar(x_encoded))

        #std = torch.exp(0.5 * logvar)
        #eps = torch.randn_like(std).to(device)
        #z = mu + eps * std

        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).mean()

        #self.kl = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())) /BATCHSIZE
        return z

class Decoder(nn.Module):
    def __init__(self, image_channels, latent_dim):
        super(Decoder, self).__init__()

        # Decoder: Latent + Pose -> Image
        self.decoder_input = nn.Linear(latent_dim , 256 * 16 * 16)  # latent + pose_dim = 99
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # (B, 128, 32, 32)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # (B, 64, 64, 64)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # (B, 32, 128, 128)
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=4, stride=2, padding=1),  # (B, 3, 256, 256)
            nn.Sigmoid()  # Output between 0 and 1
        )

    def forward(self, z):

        # Combine latent vector and pose
        # Decode
        x_reconstructed = self.decoder_input(z)
        x_reconstructed = x_reconstructed.view(-1, 256, 16, 16)  # Reshape for ConvTranspose
        x_reconstructed = self.decoder(x_reconstructed)
        return x_reconstructed

class VAE(nn.Module):
    def __init__(self, image_channels=1, latent_dim=LATENT_DIM):
        super(VAE, self).__init__()
        self.encoder = VariationalEncoder(image_channels, latent_dim)
        self.decoder = Decoder(image_channels, latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

###########################################################################################################

class Local_INN(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.model = self.build_inn()


        self.trainable_parameters = [p for p in self.model.parameters() if p.requires_grad]
        for param in self.trainable_parameters:
            param.data = 0.05 * torch.randn_like(param)
        self.cond_net = self.subnet_cond(COND_OUT_DIM)
        self.vae = VAE()

    def subnet_cond(self, c_out):
        return nn.Sequential(nn.Linear(COND_DIM, 256), nn.ReLU(),
                             nn.Linear(256, COND_OUT_DIM))

    def build_inn(self):

        def subnet_fc(c_in, c_out):
            return nn.Sequential(nn.Linear(c_in, 1024 *GLOW_COUPLINGBLOCK_FACTOR), nn.ReLU(),
                                    nn.Linear(1024*GLOW_COUPLINGBLOCK_FACTOR, c_out))

        nodes = [InputNode(N_DIM, name='input')]
        cond = ConditionNode(COND_OUT_DIM, name='condition')
        for k in range(6):
            nodes.append(Node(nodes[-1],
                                GLOWCouplingBlock,
                                {'subnet_constructor': subnet_fc, 'clamp': 2.0},
                                conditions=cond,
                                name=F'coupling_{k}'))
            nodes.append(Node(nodes[-1],
                                PermuteRandom,
                                {'seed': k},
                                name=F'permute_{k}'))

        nodes.append(OutputNode(nodes[-1], name='output'))

        return ReversibleGraphNet(nodes + [cond], verbose=False).to(self.device)

    def forward(self, x, cond):
        return self.model(x, self.cond_net(cond))

    def reverse(self, y_rev, cond):
        return self.model(y_rev, self.cond_net(cond), rev=True)

import os
class Trainer():
    def __init__(self, exp_name, max_epoch, goal_loss, device,
                 initial_rl, rl_schedule_epoch, rl_schedule_gamma = 0.5, lr_method = 'step',
                 instability_recover = False,
                 instable_thresh = 3, instable_gamma = 0.5, use_mix_precision = 0):

        self.rl_schedule_epoch = rl_schedule_epoch + [max_epoch]
        self.rl_schedule_gamma = rl_schedule_gamma
        self.rl_schedule = np.ones_like(self.rl_schedule_epoch) * initial_rl
        for ind in range(len(self.rl_schedule)):
            self.rl_schedule[ind] = self.rl_schedule[ind] * rl_schedule_gamma ** (ind + 1)
        self.instable_thresh = instable_thresh
        self.instable_gamma = instable_gamma
        self.max_epoch = max_epoch
        self.goal_loss = goal_loss
        self.exp_name = exp_name
        self.device = device
        self.instability_recover = instability_recover
        self.use_mix_precision = use_mix_precision
        self.mix_precision_relax_cnt = 2
        self.lr_method = lr_method

        self.lr = initial_rl
        self.lr_schedule = initial_rl
        self.epoch = 0
        self.best_epoch_info = np.ones(7) * float('inf')
        self.prev_epoch_info = np.ones(7) * float('inf')
        self.current_schedule = 0
        self.path = 'proj_camera' + exp_name + '/'
        if not os.path.exists(self.path):
            os.mkdir(self.path)



    def is_done(self):
        if self.epoch >= self.max_epoch - 1 or \
           self.best_epoch_info[2] <= self.goal_loss and self.best_epoch_info[2] > 0: # train_reverse loss
            return True
        else:
            return False


    def get_lr(self):

        if self.lr_method == 'step':
            if self.epoch == self.rl_schedule_epoch[self.current_schedule]:
                if self.lr > self.rl_schedule[self.current_schedule]:
                    self.lr = self.rl_schedule[self.current_schedule]
                    self.current_schedule += 1
                    print('Change LR to', self.lr)
        elif self.lr_method == 'linear':
            if self.epoch == self.rl_schedule_epoch[self.current_schedule]:
                self.current_schedule += 1
            if self.lr > self.rl_schedule[self.current_schedule]:
                self.lr += (self.rl_schedule[self.current_schedule] - self.lr) / (self.rl_schedule_epoch[self.current_schedule] - self.epoch)
        elif self.lr_method == 'exponential':
            if self.epoch == 0:
                return self.lr
            factor_value = self.rl_schedule_gamma ** (1/self.rl_schedule_epoch[0])
            self.lr_schedule *= factor_value
            if self.lr > self.lr_schedule:
                self.lr = self.lr_schedule
        elif self.lr_method == 'exponential_stop':
            if self.epoch > 0 and self.epoch <= self.rl_schedule_epoch[0]:
                factor_value = self.rl_schedule_gamma ** (1/self.rl_schedule_epoch[0])
                self.lr *= factor_value

        return self.lr


    def save_model(self, model, model_epoch_info, save_name, path = ""):
        # save the model

        filename = path + self.exp_name + '_model_' + save_name + '.pt'
        print(f"saved model:{filename}")
        torch.save(model.state_dict(), filename)

        filename = path + self.exp_name + '_model_' + save_name + '.npy'
        np.save(filename, model_epoch_info)


    def load_model(self, model, model_name, path = ''):
        if path.split('.')[-1] == 'pt':
            filename = path
        else:
            filename = path + self.exp_name + '_model_' + model_name + '.pt'
        print('Load from model:', filename)
        model.load_state_dict(torch.load(filename))
        model.to(self.device)
        return model


    def get_best_merit(self, epoch_info, rank=0):
        if rank == 0:
            return epoch_info[0] + epoch_info[2]

    def step(self, model, epoch_info, step_use_mix):
        path = self.path
        self.save_model(model, epoch_info, 'last', path)
        return_text = ''
        use_mix_precision = self.use_mix_precision
        #print(f"epoch 0 and 1 is :{epoch_info[0]}, {epoch_info[1]}")

        if self.get_best_merit(self.best_epoch_info) > self.get_best_merit(epoch_info):
            self.best_epoch_info = epoch_info.copy()
            self.save_model(model, epoch_info, 'best', path)
            return_text = 'best'

        # detect instable epoch
        
        if self.instability_recover and (self.get_best_merit(epoch_info) > self.get_best_merit(self.best_epoch_info) * self.instable_thresh \
                                         or any(np.isnan(epoch_info))):
            print('Instable epoch detected.')
            print("is NAN:",any(np.isnan(epoch_info)))
            print("epoch info:", epoch_info)
            model = self.load_model(model, 'best', path)
            self.lr *= self.instable_gamma
            print('Change LR to', self.lr)
            self.epoch = int(self.best_epoch_info[3].copy()) + 1
            if any(np.isnan(epoch_info)) and step_use_mix and self.use_mix_precision:
                use_mix_precision = 0
                self.mix_precision_relax_cnt = 5
            return_text = 'instable'
        else:
            self.prev_epoch_info = epoch_info.copy()
            self.epoch += 1

        if self.mix_precision_relax_cnt > 0 and self.use_mix_precision:
            use_mix_precision = 0
            self.mix_precision_relax_cnt -= 1

        return model, return_text, use_mix_precision == 1

class PositionalEncoding_torch():
    def __init__(self, L):
        self.L = L
        self.val_list = []
        for l in range(L):
            self.val_list.append(2.0 ** l)
        self.val_list = torch.Tensor(self.val_list)[None, :].to(device)
        self.pi = torch.Tensor([3.14159265358979323846]).to(device)

    def encode(self, x):
        return torch.sin(x * self.val_list * self.pi), torch.cos(x * self.val_list * self.pi)

    def encode_even(self, x):
        return torch.sin(x * self.val_list * self.pi * 2), torch.cos(x * self.val_list * self.pi * 2)

    def batch_encode(self, batch):
        batch_encoded_list = []
        for ind in range(3):
            if ind == 2:
                encoded_ = self.encode_even(batch[:, ind, None])
            else:
                encoded_ = self.encode(batch[:, ind, None])
            batch_encoded_list.append(encoded_[0])
            batch_encoded_list.append(encoded_[1])
        batch_encoded = torch.stack(batch_encoded_list)
        batch_encoded = batch_encoded.transpose(0, 1).transpose(1, 2).reshape((batch_encoded.shape[1], self.L * batch_encoded.shape[0]))
        return batch_encoded

    def batch_decode(self, sin_value, cos_value):
        atan2_value = torch.arctan2(sin_value, cos_value) / (self.pi)
        sub_zero_inds = torch.where(atan2_value < 0)
        atan2_value[sub_zero_inds] = atan2_value[sub_zero_inds] + 1
        return atan2_value

    def batch_decode_even(self, sin_value, cos_value):
        atan2_value = torch.arctan2(sin_value, cos_value) / self.pi/2
        sub_zero_inds = torch.where(atan2_value < 0)
        atan2_value[sub_zero_inds] = atan2_value[sub_zero_inds] + 1
        atan2_value[torch.where(torch.abs(atan2_value - 1) < 0.001)] = 0
        return atan2_value


class PositionalEncoding_torch_bk():
    def __init__(self, L):
        self.L = L
        self.val_list = []
        for l in range(L):
            self.val_list.append(2.0 ** l)
        self.val_list = torch.Tensor(self.val_list)[None, :]  # Removed .to(device) here
        #print(self.val_list)
        self.pi = torch.Tensor([3.14159265358979323846])  # Removed .to(device) here

    def encode(self, x):
        device = x.device  # Dynamically move to the same device as input tensor 'x'
        val_list = self.val_list.to(device)
        pi = self.pi.to(device)
        return torch.sin(x * val_list * pi), torch.cos(x * val_list * pi)

    def encode_even(self, x):
        device = x.device
        val_list = self.val_list.to(device)
        pi = self.pi.to(device)
        return torch.sin(x * val_list * pi * 2), torch.cos(x * val_list * pi * 2)

    def batch_encode(self, batch):
        device = batch.device  # Ensure batch device consistency
        batch_encoded_list = []
        for ind in range(3):
            if ind == 2:
                encoded_ = self.encode_even(batch[:, ind, None].to(device))
            else:
                encoded_ = self.encode(batch[:, ind, None].to(device))
            batch_encoded_list.append(encoded_[0])
            batch_encoded_list.append(encoded_[1])
        batch_encoded = torch.stack(batch_encoded_list)
        batch_encoded = batch_encoded.transpose(0, 1).transpose(1, 2).reshape(
            (batch_encoded.shape[1], self.L * batch_encoded.shape[0])
        )
        return batch_encoded

    def batch_decode(self, sin_value, cos_value):
        device = sin_value.device  # Ensure consistent device
        pi = self.pi.to(device)
        atan2_value = torch.arctan2(sin_value, cos_value) / pi
        sub_zero_inds = torch.where(atan2_value < 0)
        atan2_value[sub_zero_inds] = atan2_value[sub_zero_inds] + 1
        return atan2_value

    def batch_decode_even(self, sin_value, cos_value):
        device = sin_value.device
        pi = self.pi.to(device)
        atan2_value = torch.arctan2(sin_value, cos_value) /pi
        sub_zero_inds = torch.where(atan2_value < 0)
        atan2_value[sub_zero_inds] = atan2_value[sub_zero_inds] + 2
        #atan2_value[torch.where(torch.abs(atan2_value - 1) < 0.001)] = 0
        return atan2_value/2

class DataProcessor():
    def __init__(self) -> None:
        pass

    def two_pi_warp(self, angles):
        twp_pi = 2 * np.pi
        return (angles + twp_pi) % (twp_pi)
        # twp_pi = 2 * np.pi
        # if angle > twp_pi:
        #     return angle - twp_pi
        # elif angle < 0:
        #     return angle + twp_pi
        # else:
        #     return angle

    def data_normalize(self, data):
        data_min = np.min(data)
        data = data - data_min
        data_max = np.max(data)
        data = data / data_max
        return data, [data_max, data_min]

    def runtime_normalize(self, data, params):
        return (data - params[1]) / params[0]

    def de_normalize(self, data, params):
        return data * params[0] + params[1]

def main():
    from tqdm import trange
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter('proj_camera/' + "exp_tensorboard")

    train_loader = train_dataloader
    test_loader = test_dataloader

    l1_loss = torch.nn.L1Loss()

    p_encoding_t_17 = PositionalEncoding_torch(ENCODING_LENGTH)
    p_encoding_t = PositionalEncoding_torch(1)

    model = Local_INN(device)

    #model.load_state_dict(torch.load(partially_trained_model_path, map_location=device))
    #print("model loaded with pre trained weights")

    model.to(device)

    current_lr = LR
    optimizer = torch.optim.Adam(model.trainable_parameters, lr=current_lr)
    optimizer.add_param_group({"params": model.cond_net.parameters(), "lr": current_lr})
    optimizer.add_param_group({"params": model.vae.encoder.parameters(), "lr": current_lr})
    optimizer.add_param_group({"params": model.vae.decoder.parameters(), "lr": current_lr})
    n_hypo = 20
    epoch_time = 0
    mix_precision = False
    scaler = GradScaler(enabled=mix_precision)

    trainer = Trainer("local_INN", 700, 0.0001, device,
                      LR, [300], 0.05, 'exponential',
                      INSTABILITY_RECOVER, 3, 0.99, USE_MIX_PRECISION_TRAINING)

    while(not trainer.is_done()):
        epoch = trainer.epoch
        epoch_info = np.zeros(7)
        epoch_info[3] = epoch
        epoch_time_start = time.time()

        if USE_MIX_PRECISION_TRAINING:
            scaler = GradScaler(enabled=mix_precision)

        trainer_lr = trainer.get_lr()
        if trainer_lr != current_lr:
            current_lr = trainer_lr
            optimizer.param_groups[0]['lr'] = current_lr
            optimizer.param_groups[1]['lr'] = current_lr
            optimizer.param_groups[2]['lr'] = current_lr
            optimizer.param_groups[3]['lr'] = current_lr

        model.train()
        model.vae.encoder.train()
        model.vae.decoder.train()
        
        transf_time =0

        for (images, poses, cond_poses) in train_loader:
            st_ld =time.time()
            images = images.to(device)
            poses = poses.to(device)
            cond_poses = cond_poses.to(device)
            transf_time += time.time()-st_ld
            #print("gpu transfer time is", time.time()-st_ld)

            optimizer.zero_grad()
            with autocast(enabled=mix_precision):

                encoded_poses = p_encoding_t_17.batch_encode(poses)  #size is 102
                #ecoded_poses_for_decoder = p_encoding_t.batch_encode(poses)  #size is 6
                #ecoded_poses_for_decoder.to(device)

                encoded_cond_poses = p_encoding_t.batch_encode(cond_poses)

                #print("encoded poses are:", encoded_poses)

                x_hat_gt = encoded_poses #data[:, :60]  # position encoding ground truth pose
                #x_hat_gt = x_hat_gt.to(device)
                cond = encoded_cond_poses#data[:, 60+270:60+270+6] # position encoding of conditioned pose
                #cond.to(device)
                y_gt =  images #.to(device) #data[:, 60:60+270]  # lidar normalized data

                y_hat_vae = torch.zeros_like(x_hat_gt, device=device)
                y_hat_vae[:, :-6] = model.vae.encoder.forward(y_gt)
                y_hat_inn, _ = model(x_hat_gt, cond)
                y_inn = model.vae.decoder.forward(y_hat_inn[:, :-6])

                #nan_mask = torch.isnan(poses)
                #has_nan = torch.any(nan_mask)
                #if(has_nan):
                   #print("poses Tensor contains NaN:", has_nan.item())
                #print("y_inn values are", y_inn)

                vae_kl_loss = model.vae.encoder.kl * 0.0001
                inn_recon_loss = l1_loss(y_inn, y_gt)
                #print(f"max:{torch.max(y_inn)}, min:{torch.min(y_inn)}")
                #print(f"gt max:{torch.max(y_gt)}, min:{torch.min(y_gt)}")
                #print(f"y_inn:{y_inn}, y_gt:{y_gt}")
                #print(f"inn-recon-loss:{inn_recon_loss}")
                y_hat_inn_loss = l1_loss(y_hat_inn[:, :-6], y_hat_vae[:, :-6])
                #print(f"all losses: {vae_kl_loss} + {inn_recon_loss} + {y_hat_inn_loss}")

                #print("losses: vae_kl_loss , inn_recon_loss , y_hat_inn_loss", vae_kl_loss.item() ,inn_recon_loss.item() , y_hat_inn_loss.item())

                loss_forward = vae_kl_loss + inn_recon_loss + y_hat_inn_loss
                epoch_info[0] += loss_forward.item()
                #print(f"epoch_0: {loss_forward.item()}")

                epoch_info[1] += inn_recon_loss.item()
            scaler.scale(loss_forward).backward(retain_graph=True)
            with autocast(enabled=mix_precision):
                y_hat_vae[:, -6:] = 0
                x_hat_0, _ = model.reverse(y_hat_vae, cond)
                #loss_reverse = l1_loss(x_hat_0[:, :12], x_hat_gt[:, :12])  
                loss_reverse = l1_loss(x_hat_0[:, [2, 3, 8, 9]], x_hat_gt[:, [2, 3, 8, 9]]) 

                batch_size = y_gt.shape[0]
                z_samples = torch.cuda.FloatTensor(n_hypo, batch_size, 6, device=device).normal_(0., 1.)
                y_hat = y_hat_vae[None, :, :LATENT_DIM].repeat(n_hypo, 1, 1)
                y_hat_z_samples = torch.cat((y_hat, z_samples), dim=2).view(-1, 6*ENCODING_LENGTH)
                cond = cond[None].repeat(n_hypo, 1, 1).view(-1, COND_DIM)
                x_hat_i = model.reverse(y_hat_z_samples, cond)[0].view(n_hypo, batch_size, 6*ENCODING_LENGTH)
                #x_hat_i_loss = torch.mean(torch.min(torch.mean(torch.abs(x_hat_i[:, :, :12] - x_hat_gt[:, :12]), dim=2), dim=0)[0])

                x_hat_i_loss = torch.mean(torch.min(torch.mean(torch.abs(x_hat_i[:, :, [2, 3, 8, 9]] - x_hat_gt[:, [2, 3, 8, 9]]), dim=2), dim=0)[0])
                loss_reverse += x_hat_i_loss
                epoch_info[2] += loss_reverse.item()
                #print(f"epoch_0: {loss_reverse.item()}")

            scaler.scale(loss_reverse).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 8)

            scaler.step(optimizer)
            scaler.update()
        print("gpu transfer time is",transf_time)
        epoch_info[:3] /= len(train_loader)

        x_low = x_min
        x_high = x_max
        x_length = x_high- x_low
        y_low = y_min
        y_high = y_max
        y_length = y_high -y_low

        dp = DataProcessor()
        # testing
        model.eval()
        model.vae.encoder.eval()
        model.vae.decoder.eval()
        epoch_posit_err = []
        epoch_orient_err = []
        with torch.no_grad():
            for (images, poses, cond_poses) in test_loader:
                images = images.to(device)
                poses = poses.to(device)
                cond_poses = cond_poses.to(device)
                with autocast(enabled=mix_precision):

                    encoded_poses = p_encoding_t_17.batch_encode(poses)

                    encoded_cond_poses = p_encoding_t.batch_encode(cond_poses)

                    x_hat_gt = encoded_poses #data[:, :60]
                    #x_hat_gt =x_hat_gt.to(device)
                    x_gt = poses #data[:, 60+270+6:60+270+6+3]
                    #x_gt =x_gt.to(device)
                    cond = encoded_cond_poses #data[:, 60+270:60+270+6]
                    #cond = cond.to(device)
                    y_gt = images #.to(device) #data[:, 60:60+270]
                    # y_gt += y_gt * torch.zeros_like(y_gt, device=device).normal_(0., SCAN_NOISE)


                    y_hat_vae = torch.zeros_like(x_hat_gt, device=device)
                    y_hat_vae[:, :-6] = model.vae.encoder.forward(y_gt)
                    x_hat_0, _ = model.reverse(y_hat_vae, cond)

                
                    result_posit = torch.zeros((x_hat_gt.shape[0], 2)).to(device)

                    result_posit[:, 0] = dp.de_normalize(p_encoding_t_17.batch_decode(x_hat_0[:, 0], x_hat_0[:, 1]), [ x_length, x_low])
                    result_posit[:, 1] = dp.de_normalize(p_encoding_t_17.batch_decode(x_hat_0[:, 2], x_hat_0[:, 3]), [ y_length, y_low])
                    gt_posit = torch.zeros((x_hat_gt.shape[0], 2), device=device) #torch.zeros((x_hat_gt.shape[0], 2)).to(device)
                    gt_posit[:, 0] = dp.de_normalize(x_gt[:, 0], [ x_length, x_low]) 
                    gt_posit[:, 1] = dp.de_normalize(x_gt[:, 1], [ y_length, y_low])
                    epoch_posit_err.append(torch.median(torch.norm(result_posit[:, 1] - gt_posit[:, 1])))




                    result_angles = torch.zeros((x_hat_gt.shape[0]), device=device)  #torch.zeros((x_hat_gt.shape[0])).to(device)
                    result_angles = p_encoding_t_17.batch_decode_even(x_hat_0[:, 4], x_hat_0[:, 5])

                    # check if angle should be converted to 0 to 2pi

                    #count_gt1 =torch.sum(result_angles>1).item()
                    #count_lt0 =torch.sum(result_angles<0).item()
                    #max_val = torch.max(result_angles).item()
                    #min_val = torch.min(result_angles).item()


                    #print("yaw angle lessthan 0 and greater than 1, max and min:, ",count_lt0, count_gt1, max_val, min_val )

                    orient_err = torch.abs(result_angles - x_gt[:, 2]) * 2 * 3.14159 
                    orient_err2 = orient_err  - orient_err
                    orient_err = torch.min(orient_err2, orient_err2)  #torch.min(orient_err2, orient_err)
                    epoch_orient_err.append(torch.median(orient_err))

        epoch_time = (time.time() - epoch_time_start)
        print("Epoch time =", epoch_time)
        remaining_time = (trainer.max_epoch - epoch) * epoch_time / 3600

        epoch_info[4] = torch.median(torch.stack(epoch_posit_err))
        epoch_info[5] = torch.median(torch.stack(epoch_orient_err))
        print("step executed")

        model, return_text, mix_precision = trainer.step(model, epoch_info, mix_precision)
        if return_text == 'instable':
            optimizer = torch.optim.Adam(model.trainable_parameters, lr=current_lr)
            optimizer.add_param_group({"params": model.cond_net.parameters(), "lr": current_lr})
            optimizer.add_param_group({"params": model.vae.encoder.parameters(), "lr": current_lr})
            optimizer.add_param_group({"params": model.vae.decoder.parameters(), "lr": current_lr})


        writer.add_scalar("INN/0_forward", epoch_info[0], epoch)
        writer.add_scalar("INN/1_recon", epoch_info[1], epoch)
        writer.add_scalar("INN/2_reverse", epoch_info[2], epoch)
        writer.add_scalar("INN/3_T_pos", epoch_info[4], epoch)
        writer.add_scalar("INN/4_T_yaw", epoch_info[5], epoch)
        writer.add_scalar("INN/5_LR", current_lr, epoch)
        writer.add_scalar("INN/6_remaining(h)", remaining_time, epoch)

        text_print = "Epoch {:d}".format(epoch) + \
            ' |f {:.5f}'.format(epoch_info[0]) + \
            ' |recon {:.5f}'.format(epoch_info[1]) + \
            ' |r {:.5f}'.format(epoch_info[2]) + \
            ' |T_pos {:.5f}'.format(epoch_info[4]) + \
            ' |T_yaw {:.5f}'.format(epoch_info[5]) + \
            ' |h_left {:.1f}'.format(remaining_time) + \
            ' | ' + return_text
        print(text_print)
        with open('proj_camera/' +'log.txt', "a") as tgt:
            tgt.writelines(text_print + '\n')
    writer.flush()


if __name__ == '__main__':
    #multiprocessing.set_start_method("spawn")
    main()
    print("completed")
