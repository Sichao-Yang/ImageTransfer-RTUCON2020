'''
this script is made to test the image-transfer inference time
for a given image batch
user input:
root - the root folder for trained model, image data and result storage
modelname - the model used for inference
'''
from __future__ import print_function
import argparse
import os

import torch
import torchvision.transforms as transforms
import time
from utils import is_image_file, load_img, save_img

# User settings
root = r"F:\2. Work Project\2020_FEM_ImageTransfer\00. Codes\0. Basic ImageTransfer\0. ModelDataResult\4. speed comparison"
modelname = "/model/netG_model_epoch_80.pth"
# choice gpu or cpu
device = torch.device('cpu')
# generate working folders and make sure you put the necessary model and data into folder
if not(os.path.isdir(root+'/model')):	# check if there is directory
	os.makedirs(root+'/model')
if not(os.path.isdir(root+'/data')):	# check if there is directory
	os.makedirs(root+'/data')
if not(os.path.isdir(root+'/result')):	# check if there is directory
	os.makedirs(root+'/result')

# load generater, remember two helper function is needed - utils, networks
net_g = torch.load(root+modelname, map_location='cpu')
# print(net_g)	# check network info
# load dataset
image_dir = root+"/data/a/"
# get images name
image_filenames = [x for x in os.listdir(image_dir) if is_image_file(x)]
# image process to make them network compatible
transform_list = [transforms.ToTensor(),
                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
transform = transforms.Compose(transform_list)
# start inference, clocking
start_time = time.time()
for image_name in image_filenames:
    img = load_img(image_dir + image_name)
    img = transform(img)
    input = img.unsqueeze(0).cpu()
    out = net_g(input)
    out_img = out.detach().squeeze(0).cpu()
    save_img(out_img, root+"/result/{}".format(image_name))
print('time spend {}'.format(time.time()-start_time))