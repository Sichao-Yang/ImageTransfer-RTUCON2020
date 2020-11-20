# this script is to transfer geometry images to flux density plots
import os
import torch
import torchvision.transforms as transforms
import time
from utils import is_image_file, load_img, save_img

# gpu or cpu
device = torch.device('gpu') if torch.cuda.is_available() else torch.device('cpu')
# model path
path = os.path.dirname(os.path.dirname(os.getcwd()))
model_path = os.path.join(path,r'0. Basic ImageTransfer\0. ModelDataResult\3. trainedModel\FEA_p1_2004\netG_model_epoch_80.pth')
# load generator
net_g = torch.load(model_path) if torch.cuda.is_available() else torch.load(model_path, map_location='cpu')
# 设置读取图片地址
path = os.path.dirname(os.getcwd())
image_dir = os.path.join(path,"checkpoint\ITimage")
# result path
result_path = os.path.join(os.path.dirname(image_dir),"ITtransfered")
if not os.path.exists(result_path):
	os.makedirs(result_path)
# 抓出图片名称
image_filenames = [x for x in os.listdir(image_dir) if is_image_file(x)]
# 图片处理，不需要crop
transform_list = [transforms.ToTensor(),
                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
transform = transforms.Compose(transform_list)
# 模型计算用device，模型输出成图片用cpu了
start_time = time.time()
for image_name in image_filenames:
    img = load_img(os.path.join(image_dir,image_name))
    img = transform(img)
    input = img.unsqueeze(0)
    out = net_g(input)
    out_img = out.detach().squeeze(0)
    # save images
    save_img(out_img, os.path.join(result_path,image_name))
