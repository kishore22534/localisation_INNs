#!/usr/bin/env python
# Libraries
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
import os
import re
from PIL import Image

from sklearn.model_selection import train_test_split

# Pytorch
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

# FrEIA imports
import FrEIA.framework as Ff
import FrEIA.modules as Fm
from FrEIA.framework import InputNode, OutputNode, Node, ReversibleGraphNet, ConditionNode
from FrEIA.modules import GLOWCouplingBlock, PermuteRandom, RNVPCouplingBlock

# cuda
from torch.cuda.amp import GradScaler, autocast

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import csv



# Folder containing the images
#folder_path = "/kaggle/input/test-images-new/test_pose_images"
image_counter = 1 

tained_model_weights_path ='result_55K_sampled_uniformly_from2.6l_data_singlemodel/proj_cameralocal_INN/local_INN_model_best.pt'

INFERENCE_SAMPLES = 40
ENCODING_LENGTH =20
LATENT_DIM = 6*ENCODING_LENGTH -6  #96
POSE_DIM = 6
IMAGE_CHANNELS = 1  

GLOW_COUPLINGBLOCK_FACTOR =1

BATCHSIZE = int(32)
N_DIM = int(LATENT_DIM+6)
COND_DIM = 6
COND_OUT_DIM = 12
LR = 5e-4

INSTABILITY_RECOVER = 1
USE_MIX_PRECISION_TRAINING = 0

seed_value = 42
np.random.seed(seed_value)

mean = 0
std_dev = 0.5
std_dev_theta = 11.5

lower_bound = -2
upper_bound = 2

theta_lower_bound = -57.3
theta_upper_bound = 57.3

# # Min and max for x, y, yaw
# x_min= -7.999999999999999
# x_max= 8.600000000000016
# y_min = -5
# y_max = 2.6

# Min and max for x, y, yaw
x_min= -8.1
x_max= 8.7
y_min = -5.0
y_max = 2.6

device='cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)
print(device)


class PositionalEncoding():
    def __init__(self, L):
        self.L = L
        self.val_list = []
        for l in range(L):
            self.val_list.append(2.0 ** l)
        self.val_list = np.array(self.val_list)

    def encode(self, x):
        return np.sin(self.val_list * np.pi * x), np.cos(self.val_list * np.pi * x)

    def encode_even(self, x):
        return np.sin(self.val_list * np.pi * 2 * x), np.cos(self.val_list * np.pi * 2 * x)

    def pose_encode(self, pose):
        encoded_list = []
        
        for ind, value in enumerate(pose):
            if ind == 2:  # Encode theta using encode_even
                sin_enc, cos_enc = self.encode_even(value)
            else:  # Encode x and y using encode
                sin_enc, cos_enc = self.encode(value)
            
            # Append sine and cosine encodings to the list
            encoded_list.append(sin_enc)
            encoded_list.append(cos_enc)
        
        encoded_list = np.array(encoded_list)
        encoded_list = encoded_list.flatten('F')
        
        return encoded_list
    
    def decode(self, sin_value, cos_value):
        atan2_value = np.arctan2(sin_value, cos_value) / (np.pi)
        if np.isscalar(atan2_value) == 1:
            if atan2_value > 0:
                return atan2_value
            else:
                return 1 + atan2_value
        else:
            atan2_value[np.where(atan2_value < 0)] = atan2_value[np.where(atan2_value < 0)] + 1
            return atan2_value

    def decode_even(self, sin_value, cos_value):
        atan2_value = np.arctan2(sin_value, cos_value) / np.pi/2
        if np.isscalar(atan2_value) == 1:
            if atan2_value < 0:
                atan2_value = 1 + atan2_value
            if np.abs(atan2_value - 1) < 0.001:
                atan2_value = 0
            #print("here1")
        else:
            atan2_value[np.where(atan2_value < 0)] = atan2_value[np.where(atan2_value < 0)] + 1
            atan2_value[np.where(np.abs(atan2_value - 1) < 0.001)] = 0
            #print("here2")
        return atan2_value
        
    def decode_even_exp(self, sin_value, cos_value):
        atan2_value = np.arctan2(sin_value, cos_value) / np.pi
        if np.isscalar(atan2_value) == 1:
            if atan2_value < 0:
                atan2_value = 2 + atan2_value
            #if np.abs(atan2_value - 1) < 0.001:
                #atan2_value = 0
            #print("here1")
        else:
            atan2_value[np.where(atan2_value < 0)] = atan2_value[np.where(atan2_value < 0)] + 2
            atan2_value[np.where(np.abs(atan2_value - 1) < 0.001)] = 0
            #print("here2")
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


print_count =1
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
        self.kl = 0

    def forward(self, x, random_nums):
        global print_count
        #print("x shape is :", x.shape, type(x))
        x_encoded = self.encoder(x)
        ##########################################
        # if print_count %100 ==0:
        #     rospy.loginfo("####################################")
        #     #rospy.loginfo(f"x_encoded values: {x_encoded.detach().cpu().numpy().tolist()}")
        #     rospy.loginfo(f"shape is: {tuple(x_encoded.shape)}")
        #     rospy.loginfo("")  # Blank line

            # Open a file in append mode
            # file_path = "encoded_values.txt"
            # with open(file_path, "a") as f:
            #     # Write the x_encoded values to the file
            #     f.write("x_encoded values:\n")
            #     f.write(str(x_encoded.detach().cpu().numpy().tolist()))  # Convert to string and write
            #     f.write("\n\n")  # Blank line after values

        #print_count +=1
        #############################
        #print("x_encode.shape", x_encoded.shape)
        mu = self.fc_mu(x_encoded)
        sigma = torch.exp(self.fc_logvar(x_encoded))

        # Create CSV writers if this is the first print
        mu_file = 'encoded_mu_values.csv'
        sigma_file = 'encoded_sigma_values.csv'

        mu_np = mu.detach().cpu().numpy()
        sigma_np = sigma.detach().cpu().numpy()

        # Append mode
        with open(mu_file, 'a', newline='') as f_mu, \
            open(sigma_file, 'a', newline='') as f_sigma:
            
            mu_writer = csv.writer(f_mu)
            sigma_writer = csv.writer(f_sigma)

            for mu_row, sigma_row in zip(mu_np, sigma_np):
                mu_writer.writerow(mu_row)
                sigma_writer.writerow(sigma_row)

        # with open('encoded_sigma_values.txt', 'a') as f:
        #     f.write(str(print_count))
        #     f.write("\n\n")
        #     #f.write("sigma shape: {}\n".format(tuple(sigma.shape)))
        #     f.write("sigma values:\n")
        #     f.write(str(sigma.detach().cpu().numpy().tolist()))
        #     f.write("\n\n")


        # with open('encoded_mu_values.txt', 'a') as f:
        #     f.write(str(print_count))
        #     f.write("\n\n")
        #     #f.write("mu shape: {}\n".format(tuple(mu.shape)))
        #     f.write("mu values:\n")
        #     f.write(str(mu.detach().cpu().numpy().tolist()))
        #     f.write("\n\n")

        # print_count +=1
        #std = torch.exp(0.5 * logvar)
        #eps = torch.randn_like(std).to(device)
        #z = mu + eps * std

        z = mu + sigma*random_nums.to(device)
        #self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).mean()

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
        print("here VAE", image_channels, latent_dim)
        self.encoder = VariationalEncoder(image_channels, latent_dim)
        self.decoder = Decoder(image_channels, latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
    
class Local_INN(nn.Module):
    def __init__(self):
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
                             nn.Linear(256, c_out))

    def build_inn(self):
        def subnet_fc(c_in, c_out):
            return nn.Sequential(nn.Linear(c_in, 1024*GLOW_COUPLINGBLOCK_FACTOR), nn.ReLU(),
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

class Local_INN_Reverse(nn.Module):
    def __init__(self, sample_num):
        super().__init__()
        self.model = Local_INN()
        self.sample_num = sample_num

    def forward(self, scan_t, encoded_cond_t, random_nums1, random_nums2):
        sample_num = self.sample_num
        encode_scan = torch.zeros((sample_num, N_DIM)).to(torch.device(device))
        encode_scan[:, :LATENT_DIM] = self.model.vae.encoder.forward(scan_t.unsqueeze(1), random_nums1) 
        encode_scan[1:, LATENT_DIM:] = random_nums2
        # copying condition position 20 times i.e shape 20,6
        #print("dimension of repeat", encoded_cond_t.shape)
        encoded_cond = encoded_cond_t[None].repeat(sample_num, 1).view(-1, COND_DIM)
        encoded_result = self.model.reverse(encode_scan.to(device), encoded_cond.to(device))[0]
        return encoded_result

class Local_INN_TRT_Runtime():
    def __init__(self, sample_num) -> None:
        self.data_proc = DataProcessor()

        self.p_encoding_c = PositionalEncoding(L = 1)
        self.p_encoding = PositionalEncoding(L = ENCODING_LENGTH)

        self.sample_num = sample_num
        self.device = device

        self.local_inn_reverse = Local_INN_Reverse(self.sample_num)

        model_path = tained_model_weights_path

        current_directory = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_directory, model_path)



        self.local_inn_reverse.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.local_inn_reverse.to(self.device)
        self.local_inn_reverse.eval()
        print('Model loaded.')

    def reverse(self, img_tensor, prev_state):
        sample_num = self.sample_num  
        #print("prev state", prev_state)

        prev_state_normal = np.array(prev_state).copy()
        encoded_cond = torch.from_numpy(prev_state_normal).type('torch.FloatTensor').to(self.device)
        scan_t = img_tensor.to(self.device)
        
        random_nums1 = np.random.default_rng().normal(size=(sample_num, LATENT_DIM))
        random_nums2 = np.random.default_rng().normal(size=(sample_num-1, 6)) 
        random_nums1 = torch.from_numpy(random_nums1).type('torch.FloatTensor').to(self.device)
        random_nums2 = torch.from_numpy(random_nums2).type('torch.FloatTensor').to(self.device)

        #print("hhhhhhh",encoded_cond.shape, encoded_cond )

        encoded_result = self.local_inn_reverse(scan_t, encoded_cond, random_nums1, random_nums2)
        encoded_result = encoded_result.cpu().detach().numpy()

        ## decode and de-normalize the results
        results = np.zeros([sample_num, 3])
        results[:, 0] = self.p_encoding.decode(encoded_result[:, 0], encoded_result[:, 1])
        results[:, 1] = self.p_encoding.decode(encoded_result[:, 2], encoded_result[:, 3])
        results[:, 2] = self.p_encoding.decode_even(encoded_result[:, 4], encoded_result[:, 5])

        results[:, 0] = self.data_proc.de_normalize(results[:, 0], [ x_max - x_min, x_min])
        results[:, 1] = self.data_proc.de_normalize(results[:, 1], [ y_max - y_min, y_min])
        results[:, 2] = results[:, 2]*2*np.pi


        ## find the average
        #if 1:
            #result = np.zeros(3)
#             average_angle = np.arctan2(np.mean(np.sin(results[:, 2])), np.mean(np.cos(results[:, 2])))
#             if average_angle < 0:
#                 average_angle += np.pi * 2
#             result[2] = average_angle
            #angles =results[:, 2] # np.arctan2(np.sin(results[:, 2]), np.cos(results[:, 2]))
            #angles[angles<0] += np.pi * 2
            #result[2] = np.median(angles, axis=0)
            #if( result[2] <0):
                #result[2] += 2 * np.pi
            #result = np.median(results, axis=0)
        if 1:
            result = np.zeros(3)
            average_angle = np.arctan2(np.median(np.sin(results[:, 2])), np.median(np.cos(results[:, 2])))
            #average_angle = np.arctan2(np.mean(np.sin(results[:, 2])), np.mean(np.cos(results[:, 2])))
            #average_angle = -average_angle  #temporary fix
            if average_angle < 0:
                average_angle += np.pi * 2
            result[2] = average_angle *180/np.pi
            result[:2] = np.median(results[:, :2], axis=0)
        else:
            result = results[0]

        return result, results

# Min-Max Normalization function
def min_max_normalize_2(data, min_value, max_value):
    return (data - min_value) / (max_value - min_value)

# def normalize_and_condition_prev_pose(prev_pose_value): 
        
#     normalized_prev_x = min_max_normalize_2(prev_pose_value[0], x_min, x_max)
#     normalized_prev_y = min_max_normalize_2(prev_pose_value[1], y_min, y_max)
#     normalized_prev_yaw = prev_pose_value[2] /360.0

#     normalized_prev_pose = [normalized_prev_x, normalized_prev_y, normalized_prev_yaw]
    
#     #put prev poses into bins
#     cond_pose_np = np.array(normalized_prev_pose)
#     cond_pose_np = np.floor(cond_pose_np * 10.0) / 10.0
#     normalized_cond_pose_values = cond_pose_np.tolist()

#     return normalized_cond_pose_values

#####HERE#######



local_inn = Local_INN_TRT_Runtime(INFERENCE_SAMPLES)
bridge = CvBridge()
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),  # Images are now 256x256
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

image_id =44

def predict_from_model(prev_pose):
    
    #local_inn = Local_INN_TRT_Runtime(40)
    #prev_pos = path_data[0,60+270:60+270+6]
    
    # image_msg =  rospy.wait_for_message('/camera/camera', Image, timeout=2.5)
    # cv_image = bridge.imgmsg_to_cv2(image_msg, "bgr8")   

    ########################
    # global image_counter 
    # # Save image
    # filename = f"{image_counter}.jpg"
    # cv2.imwrite(filename, cv_image)
    # image_counter += 1
    ##############################
    global image_id
    #image_folder = '/home/siva/scene_clutter_experiment/changed_imgs_sift_removed_radius_0.5'
    image_folder = '/home/siva/scene_clutter_experiment/original_images'
    #image_folder = '/home/siva/scene_clutter_experiment/changed_imgs_sift_avgregion_radius_0.5'
    #image_folder = '/home/siva/scene_clutter_experiment/scen_change_Z_purturbations'


    image_path = os.path.join(image_folder, f"{image_id}.jpg")
    cv_image = cv2.imread(image_path)
    image_id = image_id +1
    if cv_image is None:
        print(f"ERRORRRRRRRRRRRRR---Failed to load image {image_path}")


    normalised_img_tensor = transform(cv_image)  

    prev_pose = np.array(prev_pose)
    result_arr = np.empty((0, 3))

    #normalize the conditional data
    prev_pose[0] = (prev_pose[0]-x_min)/(x_max - x_min)
    prev_pose[1] = (prev_pose[1]-y_min)/(y_max - y_min)
    prev_pose[2] = prev_pose[2]/360  #check this again

    # divide the condition input to 1000 regions
    prev_pose = np.floor(prev_pose* 10) / 10

    # encode the prev state
    position_data_cond = prev_pose
    p2 = PositionalEncoding(1)
    encode_cond_pose =[]
    for k in range(3):
      if k == 2:
        sine_part_cond, cosine_part_cond = p2.encode_even(position_data_cond[k])
      else:
        sine_part_cond, cosine_part_cond  = p2.encode(position_data_cond[k])

      encode_cond_pose.append(sine_part_cond)
      encode_cond_pose.append(cosine_part_cond)

    encode_cond_pose = np.array(encode_cond_pose)
    encode_cond_pose = encode_cond_pose.flatten('F')

    inferred_state, inferred_states = local_inn.reverse(normalised_img_tensor, encode_cond_pose)
    cv2.destroyAllWindows()
    return inferred_state


import rospy
from local_inn.srv import pose_communication_camera, pose_communication_cameraResponse
from std_msgs.msg import Float32MultiArray

def handle_image_request(req):
    #rospy.loginfo("Received LIDAR data for pose inference")

    prev_pose = req.image_data[0:3]    
    inferred_position = predict_from_model(prev_pose)
    
    # Return the response with the inferred pose
    return pose_communication_cameraResponse(inferred_position.tolist())

def infer_pose_service():
    rospy.init_node('local_inn_server', anonymous=True)

    # Create the service server
    rospy.Service('compute_inferred_pose_service_camera', pose_communication_camera, handle_image_request)

    rospy.loginfo("pose_communication_camera Service Ready")
    rospy.spin()

if __name__ == '__main__':
    infer_pose_service()

 
