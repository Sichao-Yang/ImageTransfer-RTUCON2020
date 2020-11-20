function defaultopt = nsgaopt()
% Function: defaultopt = nsgaopt()
% Description: Create NSGA-II default options structure.
% Syntax:  opt = nsgaopt()
%         LSSSSWC, NWPU
%   Revision: 1.3  Data: 2011-07-13
%*************************************************************************


defaultopt = struct(...
... % Optimization model
    'popsize', 50,...           % population size
    'maxGen', 100,...           % maximum generation
    'numVar', 0,...             % number of design variables
    'numObj', 0,...             % number of objectives
    'numCons', 0,...            % number of constraints
    'numAdd', 0,...            % number of constraints
    'lb', [],...                % lower bound of design variables [1:numVar]
    'ub', [],...                % upper bound of design variables [1:numVar]
    'vartype', [],...           % variable data type [1:numVar]£¬1=real, 2=integer
    'objfun', @objfun,...       % objective function
... % Optimization model components' name
    'nameObj',{{'total active weight [Kg]',  '1 / torque [1/Nm]'}},...
    'nameVar',{{}},...
    'nameCons',{{}},...
... % Initialization and output
    'initfun', {{@initpop, '    '}},...         % population initialization function (use random number as default)
    'outputfuns',{{@output2file}},...   % output function
    'outputfile', '',... % output file name
    'outputInterval', 1,...             % interval of output
    'plotInterval', 1,...               % interval between two call of "plotnsga".
... % Genetic algorithm operators
    'crossover', {{'intermediate', 1.2}},...         % crossover operator (Ratio=1.2)
    'mutation', {{'gaussian',0.6, 0.8}},...          % mutation operator (scale=0.1, shrink=0.5)
    'crossoverFraction', 1, ...                 % crossover fraction (0,1) of variables of an individual,   'auto' means 2.0 / nVar
    'mutationFraction', 1,...                   % mutation fraction of variables of an individual
... % Algorithm parameters
    'useParallel', 'super',...                          % compute objective function of a population in parallel. {'yes','no'}
    'poolsize', 0,...                                % number of workers use by parallel computation, 0 = auto select.
... % R-NSGA-II parameters
    'refPoints', [],...                              % Reference point(s) used to specify preference. Each row is a reference point.
    'refWeight', [],...                              % weight factor used in the calculation of Euclidean distance
    'refUseNormDistance', 'front',...                % use normalized Euclidean distance by maximum and minumum objectives possiable. {'front','ever','no'}
    'refEpsilon', 0.001 ...                          % parameter used in epsilon-based selection strategy
);



