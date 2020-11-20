from PIL import Image
import numpy as np
import os
import pandas as pd

# get the scale bar
def initial_Bar():
    item2 = "colormap.png"
    img2 = Image.open(item2) 
    pix2 = np.array(img2)
	# drop last col, because its value == 255 and have no use
    pix3=pix2[:,:,0:3]
    return pix3[7:-7,15,:].astype('int16') 

# get flux density
def get_B(probe, ColorInd):
    probe=probe[0:3].astype('int16')
    diff = ColorInd-probe
    diff_norm = np.linalg.norm(diff,axis=1)
    ind=np.where(diff_norm==np.min(diff_norm))
    length=ColorInd[:,0].shape[0]
    return ((1-ind[0]/length)*2.2)[0]


# start main
ColorInd=initial_Bar()
# probe location
x = 165;y = 145;
# path for conversion data
path = 'dataset\\FEA_p1_2004\\test\\b'

# get flux density for all files in path directary
dirs = os.listdir(path)
B_sum=[]
for item in dirs:
    img = Image.open(os.path.join(path,item))
    pix = np.array(img)
    probe=pix[y][x]
    B = get_B(probe, ColorInd)
    B_sum.append([item, B])
	
# save results
result_list = pd.DataFrame(B_sum)
result_list.to_csv('B_sum_true.csv',index=False)