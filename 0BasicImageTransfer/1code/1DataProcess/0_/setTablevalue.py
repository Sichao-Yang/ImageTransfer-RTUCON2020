#---------------------------------------------------------------------
#Name: setTablevalue.py
#Menu-en: 
#Type: Python
#Create: May 08, 2020 JSOL Corporation
#Comment-en: 
#---------------------------------------------------------------------
# -*- coding: utf-8 -*-
app = designer.GetApplication()
app.SetCurrentStudy(0)
app.View().SetCurrentCase(1)
for i in range(257):
	app.GetModel(0).GetStudy(0).GetDesignTable().SetValue(i, 3, 1000000)