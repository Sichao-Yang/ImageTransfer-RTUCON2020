%*************************************************************************
% This is made to couple with FE software
%   13-Dec-2017 sichao yang
%*************************************************************************
tic
options = nsgaopt();                    % create default options structure
mydir  = pwd;
idcs   = strfind(pwd,'\');
options.root = mydir(1:idcs(end));		 % the path for controled FEM
options.outputfile = strcat(options.root,'checkpoint\OptOut.csv');
options.inputfile = strcat(options.root,'checkpoint\FEin.csv');
options.datafile = strcat(options.root,'checkpoint\FEout.csv');
if exist(strcat(options.root,'checkpoint\ITimage'), 'dir')
    rmdir(strcat(options.root,'checkpoint\ITimage'),'s')
end
if exist(strcat(options.root,'checkpoint\ITtransfered'), 'dir')
    rmdir(strcat(options.root,'checkpoint\ITtransfered'),'s')
end
if exist(strcat(options.root,'checkpoint\ITresult'), 'dir')
    rmdir(strcat(options.root,'checkpoint\ITresult'),'s')
end
mkdir(strcat(options.root,'checkpoint\ITimage'))
mkdir(strcat(options.root,'checkpoint\ITtransfered'))
mkdir(strcat(options.root,'checkpoint\ITresult'))

options.popsize = popsize;                           % populaion size
options.maxGen  = maxGen;                         % max generation
if VPM ==1
    add = { 'magThick [mm]' 'Tooth width [mm]' 'Magnet angle [deg]' 'Rotor outer diameter [mm]'...
            'magWidth[mm]' 'Position of mag [mm]' 'CorebackWidth [mm]'...
            'Ja [A_per_m2]' 'Opt angle [deg]' 'lstk [mm]'...
            'total weight [Kg]' 'feTorque [Nm]'...
            'lamination [Kg]' 'magnet [Kg]' 'coil [Kg]' };
	options.numVar = 7;                     % number of design variables
    options.ub = ones(1,options.numVar)*4;        % upper bound of x  
else
    add = { 'Tooth width [mm]' 'CorebackWidth [mm]' 'Rotor outer diameter [mm]' 'magThick [mm]' 'Magnet angle [deg]' ...
            'Ja [A_per_m2]' 'Opt angle [deg]' 'lstk [mm]'...
            'total weight [Kg]' 'feTorque [Nm]'...
            'lamination [Kg]' 'magnet [Kg]' 'coil [Kg]' };    
    options.numVar = 5;                     % number of design variables
    options.ub = ones(1,options.numVar)*6;        % upper bound of x  
end
options.numAdd = length(add);
options.numObj = 2;                     % number of objectives
options.numCons = 0;                    % number of constraints
options.lb = zeros(1,options.numVar);               % lower bound of x 
options.vartype = repmat([2],[1,options.numVar]);   % integer number type
options.plotInterval = 1;          % interval between two calls of "plotnsga".
options.initfun(2) =  [];            % to initialise the first generation, leave empty for random distribution
% options.initfun(2) =  {'.\gen1_input.csv'};   % to initialise the first generation, with given distribution

options.objfun = handle;     % objective function handle
result = nsga2(options,add{:});    % begin the optimization!