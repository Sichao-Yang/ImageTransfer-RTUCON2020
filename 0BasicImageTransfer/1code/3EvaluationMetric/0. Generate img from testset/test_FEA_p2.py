from __future__ import print_function
import argparse
import os

import torch
import torchvision.transforms as transforms
import time
from utils import is_image_file, load_img, save_img

# Testing settings 记得设好使用的是哪个存储的模型
parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
parser.add_argument('--dataset', type=str, default='FEA_p2_2004', help='')
parser.add_argument('--direction', type=str, default='a2b', help='a2b or b2a')
parser.add_argument('--nepochs', type=int, default=40, help='saved model of which epochs')
parser.add_argument('--cuda', action='store_true', help='use cuda')
opt = parser.parse_args()
print(opt)
# 设好是用gpu还是cpu推断
device = torch.device('cpu')
# 要取的模型地址
model_path = "{}/netG_model_epoch_{}.pth".format(opt.dataset, opt.nepochs)
# 加载生成器
net_g = torch.load(model_path, map_location='cpu')
# 设置读取图片地址
if opt.direction == "a2b":
    image_dir = "{}/dataset/test_sample/a/".format(opt.dataset)
else:
    image_dir = "{}/dataset/test_sample/b/".format(opt.dataset)
# 抓出图片名称
image_filenames = [x for x in os.listdir(image_dir) if is_image_file(x)]
# 图片处理，不需要crop
transform_list = [transforms.ToTensor(),
                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
transform = transforms.Compose(transform_list)
# 模型计算用device，模型输出成图片用cpu了
start_time = time.time()
for image_name in image_filenames:
    img = load_img(image_dir + image_name)
    img = transform(img)
    input = img.unsqueeze(0).cpu()
    out = net_g(input)
    out_img = out.detach().squeeze(0).cpu()
    # 储存到输出地址
    if not os.path.exists(os.path.join("result", opt.dataset+str(opt.nepochs))):
        os.makedirs(os.path.join("result", opt.dataset+str(opt.nepochs)))
    save_img(out_img, "result/{}{}/{}".format(opt.dataset,opt.nepochs, image_name))
print('time spend {}'.format(time.time()-start_time))
