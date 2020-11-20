'''
This script is used to extract image pairs for data generation for the image-transfer network
copy and paste the code below to python script console in JMAG's Project Manager(left side GUI): 
'Study:' -> 'Post Calculation Scripts'
author: SichaoYang 20200120
'''
import os
app = designer.GetApplication()
s = app.GetCurrentStudy()
project = app.GetProjectPath()
path = os.path.dirname(os.path.abspath(project))	# get the containing directory name
case = s.GetCurrentCase() + 1		# image name defined by caseNo
if not(os.path.isdir(path+'/dataset')):	# check if there is a directory for dataset
	os.makedirs(path+'/dataset/a/')
	os.makedirs(path+'/dataset/b/')

Path_a = path + "/dataset/a/" + str(case)+".jpg"
Path_b = path + "/dataset/b/" + str(case)+".jpg"
app.View().SetContourView(0)
app.ExportImageWithSize(Path_a, 340, 256)
app.View().SetContourView(1)
app.ExportImageWithSize(Path_b, 340, 256)