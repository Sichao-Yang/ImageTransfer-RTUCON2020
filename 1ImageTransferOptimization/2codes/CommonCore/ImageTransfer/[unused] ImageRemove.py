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
path='D:/00. tools & codes/0. JMAG FEA GA/used FE model/databatch'
dirs = os.listdir(path)
for item in dirs:
	fullpath = os.path.join(path,item)      
	if os.path.isfile(fullpath):
		os.remove(fullpath)

path='D:/00. tools & codes/0. JMAG FEA GA/used FE model/ITresult'
dirs = os.listdir(path)
for item in dirs:
	fullpath = os.path.join(path,item)      
	if os.path.isfile(fullpath):
		os.remove(fullpath)

os.remove('D:/00. tools & codes/0. JMAG FEA GA/GA opt ver2_JMAG/JMAG_static_python/result.csv')