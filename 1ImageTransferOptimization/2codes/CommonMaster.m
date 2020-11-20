%% clean and start 
clc; clear; close all;
rmpath(genpath(pwd))
% add common files
folder ='CommonCore';
addpath(genpath(folder))

%% add runtime files
% folder ='1. NSGA+FEA_static';
% addpath(genpath(folder))
% handle=@VPM_objfun1;
%%%%%%%--------------------------
% folder ='2. NSGA+FEA+IT_static';
% addpath(genpath(folder))
% handle=@VPM_objfun2;
%%%%%%%--------------------------
folder ='1. NSGA+FEA_static';
addpath(genpath(folder))
handle=@SPM_objfun1;


%% call the main function
popsize=80;
maxGen=15;
VPM=0;      % flag: VPM opt or not
Caller