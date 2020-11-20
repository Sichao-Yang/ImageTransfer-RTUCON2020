function [y, cons, varino] = HEPM_objfun(x,inputfile,datafile)
cons = [];
isotest = 0;           % check the ouput without given inputs from parent files
Dri=0.030;           %[m] inner rotor diameter
p=4;                    %[-] pole pair
NoSlot=48;
ksf=0.71;             % slot filling factor used in fe   (3.75*11.2*16)/(13.5*76.7)=0.65, '76.7 is the total slot height
klam=0.97;          % lamination stack factor
Io_intval = 20;          % scan interval for opt current angle, start from 0deg end at 60deg
dlmwrite('.\temp.csv', x);

%% value related to FEM setting and power limit
% remember to check this value everytime you do opt
laref=151/1000;   %[m] stack length assumed for modelling (in FE, this number is multiplied by airgap equivalent length ratio)
Nm=5000;        %[rpm] target angular speed
fe=Nm/60*p;    %[Hz] target frequency
Power=50.e3;   %[W] target output power
T_required=Power/(fe/p*2*pi); %[Nm] target electromagnetic torque
filename=inputfile;

%% convert pop variable to FEM inputs
if isotest == 0
    varin(:,1) = x(:,1)*0.8 + 4;               % 'magThick [mm]'     
    varin(:,2) = x(:,2)*0.8 + 4;            % 'Tooth width [mm]'   
    varin(:,3) = x(:,3)*5 + 40;             % 'Magnet angle [deg]'
    varin(:,4) = x(:,4)*4 +140;            % 'Rotor outer diameter [mm]'
    varin(:,5) = x(:,5)*2 + 20;         % 'magLength[mm]'
    varin(:,6) = x(:,6)*2 + 30;        % 'Position of mag [mm]'
    varin(:,7) = x(:,7)*2 +15;        % 'CorebackWidth [mm]'
    varin(:,8) = 2.5e6;     
elseif isotest == 1  
    % test inputs for this script
    varin = [
2.5e6  5.2  6.5
2.0e6  6.0  5.5
2.0e6  6.5  5.
    ];
end
g=0.5/1000;       %[m] airgap length
Dro=varin(:,4)/1000;
Dsi=Dro+g*2;    %[m] stator inner diamter
Dso=260/1000;
Wscb=varin(:,7)/1000;  %[m] stator coreback
Htt=2.6/1000;       %[m] tooth tang depth
Hs=(Dso/2-Dro/2-g-Htt-Wscb);   %[m] slot depth
Bt=varin(:,2)/1000;        %[m] tooth width
Bs=(Dro+2*g+Hs)*pi/NoSlot - Bt;        %[m] slot width
Ja=varin(:,1);
Wm=varin(:,2)/1000; %[m] magnet width(thickness)
Hm=varin(:,5)/1000; %[m] magnet height(length)
[caseNoin,n]=size(varin);
%% varin will be reserved as a key to insert the result from JMAG back to the result position for GA evaluation
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
    varfe(repeatrow(:,2), :) = [];
end
[caseNofe,varNo]=size(varfe);


%% put varfe into inputfile and run JMAG control to get FEM result
% run it through each case to get the optimal current input angle
for n=1:caseNofe
    Ja =  varfe(n, end);
    ang = (0:Io_intval:60)/180*pi;
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
dlmwrite(filename, stat_in);

% run transient cases to get torque and loss
studyNo = 1;   % the number start from 0, this will set the study we want to work on
JMAGcontrol

% get the max current angle
data = readtable(datafile);
array = data{:,2};
D = str2double(array);
len = length(ang);
for n=1:caseNofe
    if sum(isnan(D(1:len)))
        idx = 1;    % just assign a number, the actual value doesnt matter
    else
        [val, idx]=max(D(1:len), [], 1);
    end
    Tm_ang(n,1)=ang(idx)/pi*180;
    D=D(len+1:end);
end
% get the opt current angle feed into the transient model
tran_in = cat(2, varfe, Tm_ang);
dlmwrite(filename, tran_in);
studyNo = 0;   % the number start from 0, this will set the study we want to work on
JMAGcontrol


%% grab the data
% first 3 columns are the output, then next 10 columns
% are the variables (7 system var + Ja + IoptAng)
% the output contains:
% Tfe lamloss magloss     (torque and losses)
%----------------------------------------------------
% these variables are the key to match the output with the input
% the caseNoout is the actual output case number, it will be used to check whether there is any case out of FE boundary
data = readtable(datafile);
array = data{:,2:end};
D = str2double(array);
outNo=3;
caseNoout=size(D,1);


%% make the fe output D to have the same unit as the varin
% change geometry variables, unit is mm,
% D(:,[outNo+1 outNo+2 outNo+3])=...
%     int16(D(:,[outNo+1 outNo+2 outNo+3]));
% % change Iangle from rad to deg
% D(:,outNo+1)=int16(D(:,outNo+1)/pi*180);
% % change bpm
% D(:,end)=round(D(:,end)*1000);      % the magnet width bpm
% % change 'g' from 0.0055m to 5.500mm
% Ndecimals = 1;
% f = 10.^Ndecimals; 
% D(:,outNo+2) = round(f*D(:,outNo+2)*1000)/f;
% %make sure 'g' is same type in var and D
% varin(:,2)= round(f*varin(:,2))/f;

%% map the FE results onto the out_part1 matrix (with size of varin)
% the last one is the IoptAng, is not included in varin so wont be compared
out_part1=zeros(caseNoin,outNo+varNo+1);  m=0;
for n=1:caseNoin
   if prod(n~=repeatrow(:,2))
     for m=1 : caseNoout
         if prod(varin(n,:)==D(m,outNo+1:end-1))
            out_part1(n,:)=D(m,:);
         end
     end
   end
end

%% copy the results based on info from 'repeatrow'
if repeatrow~=[0,0]
    for n=1:length(repeatrow(:,1))
        out_part1(repeatrow(n,2),:)=out_part1(repeatrow(n,1),:);
    end
end

%% PostProcess for: Torque, iron loss, magnet loss and material weights
Tfe=abs(out_part1(:,1)); %this is the electromagnetic torque converted from T=3VI/omega;
Iron_loss=out_part1(:,2);
Mag_loss=out_part1(:,3);
out_part1=out_part1(:,4:end);   %remove the prepossed FEoutputs

% convert to per unit value and get actual value for required torque
Tp=Tfe/laref;  %[Nm] electromagnetic torque, para per length
Iron_loss_pu=Iron_loss/laref;
Mag_loss_pu=Mag_loss/laref;
la=T_required./Tp;       %[m] adjusted stack length      

%% mass, cost and loss
WeightnCost;
Losses;
% objective one - active weight
y(:,1)=Qmag+Qlamnet+Qcoil;


%% check whether there is unmapped zero rows, defect results
%% if so, give infinite number to the objective function
zerorow=zeros(1,varNo+1);
for n=1:caseNoin
    if (prod(zerorow==out_part1(n,:)) | Tfe(n) <= 1 | Pfe<0| isnan(y(n,:)))
        y(n,:)=[inf inf];
    end
end

%% output data
varino=[out_part1 la*1000 Pw Pfe Pcu  y Tfe...
    Qlamnet Qmag Qcoil ];