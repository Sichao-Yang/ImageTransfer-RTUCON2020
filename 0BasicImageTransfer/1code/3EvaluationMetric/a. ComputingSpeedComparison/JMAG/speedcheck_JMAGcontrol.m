%*************************************************************************
% This script is used to control JMAG: launch, calculate and export the data
% 	22-Jun-2020 sichao yang
%*************************************************************************
%% this script is used to launch JMAG, do the calculation and export the data
root = 'F:\2. Work Project\2020_FEM_ImageTransfer\00. Codes\0. Basic ImageTransfer\0. ModelDataResult\4. speed comparison\JMAG\';
modelname = 'data_make0424.jproj';

tic
studyNo = 0;  
% Launch JMAG-Designer
% Select JMAG-Designer version and start JMAG-Designer
designer = actxserver('designer.Application.181');
% Display JMAG-Designer window
designer.Show();
app = designer;
app.Load(strcat(root,modelname));
% Run all cases - before run, makesure that all cases are uncalculated
% and cad model are correctedly linked with JMAGexpress
app.GetModel(0).GetStudy(studyNo).RunAllCases()
time = toc;