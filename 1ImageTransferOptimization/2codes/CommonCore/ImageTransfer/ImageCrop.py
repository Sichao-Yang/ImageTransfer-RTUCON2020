# this script is to transform geometry images to network compatible inputs
from PIL import Image
import os.path, sys
w = 340
h = 256
w_tar = 210
h_tar = w_tar
left = (w -w_tar)/2
top = (h-h_tar)/2
right = w - left
bottom = h - top
path = os.path.dirname(os.getcwd())
path = os.path.join(path,'checkpoint\ITimage')
dirs = os.listdir(path)
def crop():
	for item in dirs:
		fullpath = os.path.join(path,item)      
		if os.path.isfile(fullpath):
			im = Image.open(fullpath)
			f, e = os.path.splitext(fullpath)
			trans = im.crop((left, top, right, bottom)).resize([256,256])
			os.remove(fullpath)
			trans.save(f + '.jpg', quality=100)
crop()