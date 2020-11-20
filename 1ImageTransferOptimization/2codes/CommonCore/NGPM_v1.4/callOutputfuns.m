function opt = callOutputfuns(opt, state, pop, varargin)
% Function: opt = callOutputfuns(opt, state, pop, type)
% Description: Call output function(if exist).
% Parameters: 
%   type : output type.  
%       -1 = the last call (close file for instance)
%       other values(or no exist) = normal output
%
%         LSSSSWC, NWPU
%    Revision: 1.1  Data: 2011-07-13
%*************************************************************************

type = 0;   % normal output

if( ~isempty(opt.outputfuns) )
    fun = opt.outputfuns{1};
    opt = fun(opt, state, pop, varargin{:});
end


