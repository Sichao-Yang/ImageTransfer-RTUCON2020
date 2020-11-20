function opt = output2file(opt, state, pop, varargin)
% Description: Output the population 'pop' to file. The file name is
% specified by 'opt.outputfile' field.
% Parameters: 
%   type : output type.  -1 = the last call, close the opened file. 
%          others(or no exist) = normal output
%   varargin : any parameter define in the options.outputfuns cell array.
%
%         LSSSSWC, NWPU
%    Revision: 1.2  Data: 2011-07-13
%*************************************************************************
if(isempty(opt.outputfile))
    return;  % the output file name is not specified, return directly
end

if( isfield(opt, 'outputfileFID') )
    fid = opt.outputfileFID;
else
    fid = [];
end

%*************************************************************************
% 1.Open the output file and output some population info
%*************************************************************************
if( isempty(fid) )
    fid = fopen(opt.outputfile, 'w');
    if( fid == 0)
        error('NSGA2:OutputFileError', 'Can not open output file!! file name:%s', opt.outputfile);
    end
    opt.outputfileFID = fid;
    % Output some infomation
    fprintf(fid, '#NSGA2\r\n');
    fprintf(fid, 'popsize %d\n', opt.popsize);
    fprintf(fid, 'maxGen %d\n', opt.maxGen);
    fprintf(fid, 'numVar %d\n', opt.numVar);
    fprintf(fid, 'numObj %d\n', opt.numObj);
    fprintf(fid, 'numCons %d\n', opt.numCons);
    
    % Output state field names
    fprintf(fid, 'stateFieldNames\t');
    names = fieldnames(state);
    for i = 1:length(names)
        fprintf(fid, '%s\t', names{i});
    end
    fprintf(fid, '\r\n'); 
    fprintf(fid, '#end\r\n\r\n\r\n');
end

%*************************************************************************
% 3. Output population to file
%*************************************************************************
fprintf(fid, '#Generation %d / %d\r\n', state.currentGen, opt.maxGen);

% output each state field
names = fieldnames(state);
for i = 1:length(names)
    fprintf(fid, '%s\t%g\r\n', names{i}, getfield(state, names{i}));
end
fprintf(fid, '#end\r\n');
fprintf(fid, '%s,', varargin{1:end}) ;
for i = 1:opt.numVar
    fprintf(fid, 'Var%d,', i);
end
for i = 1:opt.numObj
    fprintf(fid, 'Obj%d,', i);
end
for i = 1:opt.numCons
    fprintf(fid, 'Cons%d,', i);
end
fprintf(fid, '\n');

for p = 1 : opt.popsize
    for i = 1:opt.numAdd
        fprintf(fid, '%g,', pop(p).add(i) );
    end
    for i = 1:opt.numVar
        fprintf(fid, '%g,', pop(p).var(i) );
    end
    for i = 1:opt.numObj
        fprintf(fid, '%g,', pop(p).obj(i) );
    end
    for i = 1:opt.numCons
        fprintf(fid, '%g,', pop(p).cons(i));
    end
    fprintf(fid, '\n');
end
% for p = 1 : opt.popsize

% end
fprintf(fid, '\r\n\r\n\r\n');

% 
% dlmwrite(filename2, gen,'-append','roffset',1);
% Pb=repmat([Pb],length(Hs),1);
% varino=[varin la*1000 Pc/1000 Ppm/1000 Pcu/1000 Pw/1000 Pb/1000 y Tfe];
% dlmwrite(filename2,varino,'-append');

% 
% function opt = output2file(opt, state, pop, type, varargin)
% % Description: Output the population 'pop' to file. The file name is
% % specified by 'opt.outputfile' field.
% % Parameters: 
% %   type : output type.  -1 = the last call, close the opened file. 
% %          others(or no exist) = normal output
% %   varargin : any parameter define in the options.outputfuns cell array.
% %
% %         LSSSSWC, NWPU
% %    Revision: 1.2  Data: 2011-07-13
% %*************************************************************************
% 
% 
% if(isempty(opt.outputfile))
%     return;  % the output file name is not specified, return directly
% end
% 
% if( isfield(opt, 'outputfileFID') )
%     fid = opt.outputfileFID;
% else
%     fid = [];
% end
% 
% %*************************************************************************
% % 1.Open the output file and output some population info
% %*************************************************************************
% if( isempty(fid) )
%     fid = fopen(opt.outputfile, 'w');
%     if( fid == 0)
%         error('NSGA2:OutputFileError', 'Can not open output file!! file name:%s', opt.outputfile);
%     end
%     opt.outputfileFID = fid;
%     
%     % Output some infomation
%     fprintf(fid, '#NSGA2\r\n');
% 
%     fprintf(fid, 'popsize %d\n', opt.popsize);
%     fprintf(fid, 'maxGen %d\n', opt.maxGen);
%     fprintf(fid, 'numVar %d\n', opt.numVar);
%     fprintf(fid, 'numObj %d\n', opt.numObj);
%     fprintf(fid, 'numCons %d\n', opt.numCons);
%     
%     % Output state field names
%     fprintf(fid, 'stateFieldNames\t');
%     names = fieldnames(state);
%     for i = 1:length(names)
%         fprintf(fid, '%s\t', names{i});
%     end
%     fprintf(fid, '\r\n');
%     
%     fprintf(fid, '#end\r\n\r\n\r\n');
% end
% 
% %*************************************************************************
% % 2. If this is the last call, close the output file
% %*************************************************************************
% if(type == -1)
%     fclose(fid);
%     rmfield(opt, 'outputfileFID');
%     return
% end
% 
% %*************************************************************************
% % 3. Output population to file
% %*************************************************************************
% fprintf(fid, '#Generation %d / %d\r\n', state.currentGen, opt.maxGen);
% 
% % output each state field
% names = fieldnames(state);
% for i = 1:length(names)
%     fprintf(fid, '%s\t%g\r\n', names{i}, getfield(state, names{i}));
% end
% fprintf(fid, '#end\r\n');
% 
% for i = 1:opt.numVar
%     fprintf(fid, 'Var%d\t', i);
% end
% for i = 1:opt.numObj
%     fprintf(fid, 'Obj%d\t', i);
% end
% for i = 1:opt.numCons
%     fprintf(fid, 'Cons%d\t', i);
% end
% fprintf(fid, '\r\n');
% 
% for p = 1 : opt.popsize
%     for i = 1:opt.numVar
%         fprintf(fid, '%g\t', pop(p).var(i) );
%     end
%     for i = 1:opt.numObj
%         fprintf(fid, '%g\t', pop(p).obj(i) );
%     end
%     for i = 1:opt.numCons
%         fprintf(fid, '%g\t', pop(p).cons(i));
%     end
%     fprintf(fid, '\r\n');
% end
% 
% fprintf(fid, '\r\n\r\n\r\n');
% 
% 
% 
% 


