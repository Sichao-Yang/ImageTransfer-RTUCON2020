'''
This script is made to process the data to make it compatible with the network interface
the operations included are crop, resize and shuffle
user input: dataset path, testfilNo
author: SichaoYang 20200120
'''
from PIL import Image
import os.path, sys

# dataset path
path='/home/yons/Downloads/1. IT/SPM'


w = 340			# image width defined in image extraction
h = 256
w_tar = 210		# target width
h_tar = w_tar
# crop coords:
left = (w-w_tar)/2
top = (h-h_tar)/2
right = w-left
bottom = h-top

# image crop and resize
# crop function take a target folder path, crop all files under and remove the original files
def Crop(target_dir):
	for item in os.listdir(target_dir):
		fullpath = os.path.join(target_dir,item)      
		if os.path.isfile(fullpath):
			im = Image.open(fullpath)
			f, e = os.path.splitext(fullpath)
			trans = im.crop((left, top, right, bottom)).resize([256,256])
			trans.save(f + 'trans.jpg', quality=100)
			os.remove(fullpath)

dirs = os.listdir(path)			
for Dir in dirs:
    Crop(os.path.join(path,Dir))			
			

# after crop, we randomly take images from folders to form test set
import os, random
import shutil
random.seed(a=1)
src1 = os.path.join(path,'a')
tar1 = os.path.join(path,'ta')
src2 = os.path.join(path,'b')
tar2 = os.path.join(path,'tb')
if not(os.path.isdir(tar1)):	# check if there is target directories
	os.makedirs(tar1)
	os.makedirs(tar2)

# the no of images we take is defined as a percentage in testfileNo
testfileNo = round(len(os.listdir(src1))*0.1)
for i in range(testfileNo):
    file = random.choice(os.listdir(src1))
    src = os.path.join(src1,file)
    tar = os.path.join(tar1,file)
    shutil.move(src, tar)
    src = os.path.join(src2,file)
    tar = os.path.join(tar2,file)
    shutil.move(src, tar)   
