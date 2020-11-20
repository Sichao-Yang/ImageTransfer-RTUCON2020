# this script is to check probed location's flux density against the threshold and classify images as 'pass' or 'discard'
from PIL import Image
import numpy as np
import os
import pandas as pd
from utils import is_image_file

def initial_Bar():
    item2 = r"CommonCore\ImageTransfer\colormap.png"
    img2 = Image.open(item2) 
    pix2 = np.array(img2)
    pix3=pix2[:,:,0:3]		 # last col are all 255，no use
    return pix3[7:-7,15,:].astype('int16') 

def get_B(probe, ColorInd):
    probe=probe[0:3].astype('int16')
    diff = ColorInd-probe
    diff_norm = np.linalg.norm(diff,axis=1)
    ind=np.where(diff_norm==np.min(diff_norm))
    length=ColorInd[:,0].shape[0]
    return ((1-ind[0]/length)*2.2)[0]


ColorInd=initial_Bar()
x = 170
y = 145
# path of transfered images
path = os.path.dirname(os.getcwd())
path = os.path.join(path,'checkpoint\ITtransfered')
threshold = 0.8		# flux density below this is discarded
Decision = [['imageNo', 'flux density', 'pass or not']]
image_filenames = [x for x in os.listdir(path) if is_image_file(x)]
for item in image_filenames:
	fullpath = os.path.join(path,item)
	img = Image.open(fullpath)
	pix = np.array(img)
	probe = pix[y][x]
	B = get_B(probe, ColorInd)
	# Classification function
	l = item.split('.')	
	f, e = os.path.splitext(fullpath)
	if B >= threshold:
		Decision.append([l[0], B, str(1)])
		img.save(f + '+.jpg', quality=100)
	else:
		Decision.append([l[0], B, str(0)])
		img.save(f + '-.jpg', quality=100)
	os.remove(fullpath)
	
path = os.path.dirname(path)
result_list = pd.DataFrame(Decision)
result_list.to_csv(os.path.join(path,'ITClassified.csv'),index=False,header=False)