function [pop, state] = evaluate(opt, pop, state, varargin)
% Function: [pop, state] = evaluate(opt, pop, state, varargin)
% Description: Evaluate the objective functions of each individual in the
%   population.
%
%         LSSSSWC, NWPU
%    Revision: 1.0  Data: 2011-04-20
%*************************************************************************

N = length(pop);
allTime = zeros(N, 1);  % allTime : use to calculate average evaluation times

%*************************************************************************
% Evaluate objective function in parallel
%*************************************************************************
if( strcmpi(opt.useParallel, 'yes') == 1 )
    curPoolsize = matlabpool('size');

    % There isn't opened worker process
    if(curPoolsize == 0)
        if(opt.poolsize == 0)
            matlabpool open local
        else
            matlabpool(opt.poolsize)
        end
    % Close and recreate worker process
    else
        if(opt.poolsize ~= curPoolsize)
            matlabpool close
            matlabpool(opt.poolsize)
        end
    end

    parfor i = 1:N
        fprintf('\nEvaluating the objective function... Generation: %d / %d , Individual: %d / %d \n', state.currentGen, opt.maxGen, i, N);
        [pop(i), allTime(i)] = evalIndividual(pop(i), opt.objfun, varargin{:});
    end

%*************************************************************************
% Evaluate objective function in serial
%*************************************************************************
elseif ( strcmpi(opt.useParallel, 'no') == 1 )
    for i = 1:N
        fprintf('\nEvaluating the objective function... Generation: %d / %d , Individual: %d / %d \n', state.currentGen, opt.maxGen, i, N);
        [pop(i), allTime(i)] = evalIndividual(pop(i), opt.objfun, varargin{:});
    end
    
%*************************************************************************
% Evaluate objective function as whole
%*************************************************************************   
elseif ( strcmpi(opt.useParallel, 'super') == 1 ) 
    fprintf('\nEvaluating the objective function... Generation: %d / %d ', state.currentGen, opt.maxGen);
    gen=state.currentGen;
    [pop, evalTime] = evalGen(pop, opt.objfun, opt.inputfile, opt.datafile, gen, opt.root); 
end

%*************************************************************************
% Statistics: if in serial, then sum up all pop sample's time, and divide
% its length; if in batch, then pop's total time is given by 'evalTime'
%*************************************************************************
if ( strcmpi(opt.useParallel, 'no') == 1 )
    state.avgEvalTime   = sum(allTime) / length(allTime);
elseif ( strcmpi(opt.useParallel, 'super') == 1 ) 
    state.avgEvalTime   = evalTime / length(pop);
end
state.evaluateCount = state.evaluateCount + length(pop);




function [indi, evalTime] = evalIndividual(indi, objfun, varargin)
% Function: [indi, evalTime] = evalIndividual(indi, objfun, varargin)
% Description: Evaluate one objective function.
%
%         LSSSSWC, NWPU
%    Revision: 1.1  Data: 2011-07-25
%*************************************************************************

tStart = tic;
[y, cons, add] = objfun( indi.var, varargin{:});
evalTime = toc(tStart);

% Save the objective values and constraint violations
indi.obj = y;
indi.add = add;
if( ~isempty(indi.cons) )
    idx = find( cons );
    if( ~isempty(idx) )
        indi.nViol = length(idx);
        indi.violSum = sum( abs(cons) );
    else
        indi.nViol = 0;
        indi.violSum = 0;
    end
end


function [genN, evalTime] = evalGen(genN, objfun, inputfile, datafile, gen, root)
% Description: Evaluate one generation as whole
%         SICHAO Yang
%    Revision: 1.1  Date: 2017-07-25
%*************************************************************************
var_mat=vertcat(genN.var);
tStart = tic;
[y, cons, add, time] = objfun( var_mat, inputfile, datafile, gen, root);
evalTime = toc(tStart) - time;
% Save the objective values and constraint violations
[caseNo,n]=size(y);
for n=1:caseNo
    genN(n).obj = y(n,:);
    genN(n).add = add(n,:);
    if( ~isempty(cons) )
        genN(n).cons= cons(n,:);
    end
end
for n=1:caseNo
if( ~isempty(genN(n).cons) )
    idx = find( cons(n,:) );
    if( ~isempty(idx) )
        genN(n).nViol = length(idx);
        genN(n).violSum = sum( abs(cons(n,:)) );
    else
        genN(n).nViol = 0;
        genN(n).violSum = 0;
    end
end
end