%*************************************************************************
% This is made to couple with FE software
%   13-Dec-2017 sichao yang
%*************************************************************************
clc; clear; close all;
tic

addpath(genpath(pwd))
options = nsgaopt();                    % create default options structure
root = 'E:\2. Work Project\FEM_ImageTransfer\200423-optimizationWithDL\00. tools & codes\0. JMAG FEA GA\used FE model\';
options.popsize = 4;                        % populaion size
options.maxGen  = 15;                      % max generation
add = { 'magThick [mm]' 'Tooth width [mm]' 'Magnet angle [deg]' 'Rotor outer diameter [mm]'...
        'magLength[mm]' 'Position of mag [mm]' 'CorebackWidth [mm]'...
        'Ja [A_per_m2]' 'Opt angle [deg]' 'lstk [mm]'...
        'all other losses [W]' 'iron loss [W]' 'copper loss [W]'...
        'total weight [Kg]' '-efficiency'  'feTorque [Nm]'...
        'lamination [Kg]' 'magnet [Kg]' 'coil [Kg]' };
options.numAdd = length(add);
options.outputfile = strcat(root,'output.csv');
options.inputfile = strcat(root,'input.csv');
options.datafile = strcat(root,'FEout.csv');
options.numObj = 2;                     % number of objectives
options.numVar = 7;                     % number of design variables
options.numCons = 0;                    % number of constraints
options.lb = zeros(1,options.numVar);               % lower bound of x
options.ub = [6 5 6 5 6 5 5];        % upper bound of x   2p[7 5 8 8 5 4 5 7 5 4]
options.vartype = repmat([2],[1,options.numVar]);   % integer number type
options.objfun = @HEPM_objfun;     % objective function handle
options.plotInterval = 1;          % interval between two calls of "plotnsga".
options.initfun(2) =  [];          % to initialise the first generation, leave empty for random generation
%     options.initfun(2) =  {'.\temp.csv'};
result = nsga2(options,add{:});    % begin the optimization!

toc