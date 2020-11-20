%*************************************************************************
% This is the function to calculate the NSGA objective from input
% 	01-Apr-2020 sichao yang
%*************************************************************************
function [y, cons, varino, time] = HEPM_objfun1(x, inputfile, datafile, gen, root)
time = 0;               % this function is to record the image generation time, here is set to 0
cons = [];
Io_intval = 10;       % scan interval for opt current angle
ang = [0:Io_intval:0]/180*pi;     % scan angle for current advancing
isotest = 0;            % check the ouput without given inputs from parent files
Dri=0.030;             %[m] inner rotor diameter
p=4;                      %[-] pole pair
NoSlot=48;
ksf=0.71;               % slot filling factor used in fe   (3.75*11.2*16)/(13.5*76.7)=0.65, '76.7 is the total slot height
klam=0.97;            % lamination stack factor
% dlmwrite('temp.csv', x);
[caseNoin, n]=size(x);       % get input case no.
la=150/1000*ones(caseNoin,1);   %[m] stack length used in FEM

%% A. map pop. variable to FEM inputs
%{ 
    add = { 'Tooth width [mm]' 'CorebackWidth [mm]' 'Rotor outer diameter [mm]' 'magThick [mm]' 'Magnet angle [deg]' ...
            'Ja [A_per_m2]' 'Opt angle [deg]' 'lstk [mm]'...
            'total weight [Kg]' 'feTorque [Nm]'...
            'lamination [Kg]' 'magnet [Kg]' 'coil [Kg]' };   
%}
if isotest == 0
    varin(:,1) = x(:,1)*0.5 + 4;           % 'Tooth width [mm]' 
    varin(:,2) = x(:,2)*2 + 12;            % 'CorebackWidth [mm]'
    varin(:,3) = x(:,3)*1 + 140;          % 'Rotor outer diameter [mm]'
    varin(:,4) = x(:,4)*1 +8;               % 'magThick [mm]'
    varin(:,5) = x(:,5)*1 + 28;            % 'Magnet angle [deg]'
    varin(:,6) = 2.0e6;                      % 'current density [A/mm2]'
elseif isotest == 1  
    % test inputs for this script
    varin = [
    2.5e6  5.2  6.5
    2.0e6  6.0  5.5
    2.0e6  6.5  5.
    ];
end
% get other geometry parameters for postprocess
g=1/1000;               %[m] airgap length
Dro=varin(:,3)/1000;
Dsi=Dro+g*2;           %[m] stator inner diamter
Dso=260/1000;
Wscb=varin(:,2)/1000;  %[m] stator coreback
Htt=2.6/1000;          %[m] tooth tang depth
Hs=(Dso/2-Dro/2-g-Htt-Wscb);   %[m] slot depth
Bt=varin(:,1)/1000;            %[m] tooth width
Bs=(Dro+2*g+Hs)*pi/NoSlot - Bt;        %[m] slot width
Wm=varin(:,4)/1000/2;    %[m] magnet width(thickness)   
% the 1/2 factor in Wm is to convert from one piece per pole to 2 pics per pole, becuz WeightnCost script is based on VPM
Hm=varin(:,5)/180*pi.*Dro/2;    %[m] magnet height(length)


%% B. get input variables for calculation
% varin will be reserved as a key to insert the result from JMAG back to the result position for GA evaluation
varfe=varin;      % varfe will be the input for JMAG 
%% find the repreated case in varin and delete them
k=0; % total number of repeated cases
repeatrow=[0,0];
for n=1:caseNoin
   for m=n+1:caseNoin
    if varin(m,:)==varin(n,:) & prod(m~=repeatrow(:,2))
         k=k+1;
         repeatrow(k,1)=n;
         repeatrow(k,2)=m;
    end
   end
end
if repeatrow~=[0,0]
    varfe(repeatrow(:,2), :)=[];
end
[caseNofe,varNo]=size(varfe);

%% put varfe into inputfile and run JMAG control to get FEM result
% step1: run it through each case to get torque vs current input angle
for n=1:caseNofe
    Ja =  varfe(n, end);
    Iu = Ja*sin(0+ang);
    Iv = Ja*sin(4/3*pi+ang);
    Iw = Ja*sin(2/3*pi+ang);
    if n==1
        % drop current density (included in iuvw already) and create the
        % input table for FEM
        stat_in = cat(2, repmat(varfe(n,1:end-1),length(Iw),1), Iu', Iv', Iw', ang');
    else
        update = cat(2, repmat(varfe(n,1:end-1),length(Iw),1), Iu', Iv', Iw', ang');
        stat_in = [stat_in; update];
    end
end
dlmwrite(inputfile, stat_in);
% run static cases to get torque
Switch=3;       % 0 - run Optmodel, 1 - run image-transfer model, 3 - run SPM opt
JMAGcontrol_v2(Switch, inputfile, datafile, root)   % remember to set file path in script

% step2: from torque output find the max torque inducing angle
data1 = readtable(datafile);
array = data1{:,2};
temp = str2double(array);
len = length(ang);
Tm = zeros(caseNofe,1);
Tm_ang = zeros(caseNofe,1);
% check if width of data is correct incase there is no torque output:
% 1+outNo+varNo-1+3+1:  
% caseNo(1) + torqueOut(1) + varNo(geometry 7 + Ja 1)
% - Ja(1) + 3phase current(3) + optCurrentAngle(1)
outNo=1;
if width(data1) == (1+outNo+varNo-1+3+1)
    for n=1:caseNofe
        if sum(isnan(temp(1:len)))
            idx = 1;    % can assign any idx no. for the angle of a invalid design 
        else
            [val, idx]=max(temp(1:len), [], 1);
        end
        Tm_ang(n,1)=ang(idx)/pi*180;   % max torque angle
        Tm(n)=temp(idx);
        temp=temp(len+1:end);   % remove the searched part
    end
end
% get the opt current angle and max Torque
D = cat(2, Tm, varfe, Tm_ang);


%% E. fillin out_part1 matrix (including repeated rows)
caseNoout=size(D,1);
% first 1 columns is the output Tfe, then next 9 columns
% are the variables (7 system var + Ja + IoptAng)
%----------------------------------------------------
% these variables are the key to match the output with the input
% the caseNoout is the actual output case number, it will be used to check whether there is any case out of FE boundary
out_part1=zeros(caseNoin, outNo+varNo+1);  m=0;
% step1: map D into output1
for n=1:caseNoin
   if prod(n~=repeatrow(:,2))
     for m=1 : caseNoout
         if prod(varin(n,:)==D(m,outNo+1:end-1))
            out_part1(n,:)=D(m,:);
         end
     end
   end
end
% step2: copy results based on info from 'repeatrow'
if repeatrow~=[0,0]
    for n=1:length(repeatrow(:,1))
        out_part1(repeatrow(n,2),:)=out_part1(repeatrow(n,1),:);
    end
end


%% F. PostProcess for: material weights
% objectives: 1. weight;  2. 1/torque, since we are doing minimization
WeightnCost;
y(:,1)=Qmag+Qlamnet+Qcoil;
Tfe=abs(out_part1(:,1));    %[Nm] torque output
regularizer=1;                  %this number will change the optimization weight between two objectives
y(:,2)=regularizer./Tfe;      %this is obj. 2
out_part1=out_part1(:,2:end);   %remove objective Tfe

%% G. Return
% final check whether there is unmapped zero rows, defect results
% if so, give infinite number to the objective function
zerorow=zeros(1,varNo+1);
for n=1:caseNoin
    if (prod(zerorow==out_part1(n,:)) | Tfe(n) <= 1| isnan(y(n,:)))
        y(n,:)=[inf inf];
    end
end
% output data
varino=[out_part1 la*1000  y(:,1) regularizer./(y(:,2))...
    Qlamnet Qmag Qcoil ];