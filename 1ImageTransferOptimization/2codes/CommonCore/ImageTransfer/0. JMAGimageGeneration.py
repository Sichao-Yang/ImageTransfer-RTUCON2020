#---------------------------------------------------------------------
#Name: JMAGimageGeneration.py
#Menu-en: 
#Type: Python
#Create: July 29, 2020 JSOL Corporation
#Comment-en: 
#---------------------------------------------------------------------
# -*- coding: utf-8 -*-
# this script is made to save images from JMAG
import os
 
app = designer.GetApplication()
s = app.GetCurrentStudy()
r = s.GetReport()
project = app.GetProjectPath()
path = os.path.dirname(os.path.abspath(project))
c		# move up one directory
case = s.GetCurrentCase() + 1
Path_a = path + "/checkpoint/ITimage/" + str(case)+".jpg"
app.ExportImageWithSize(Path_a, 340, 256)
