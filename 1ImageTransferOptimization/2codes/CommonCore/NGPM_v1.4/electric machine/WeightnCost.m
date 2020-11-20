%% stator sheet mass
DensityFe=7650;     %[kg/m3] Iron density
%% the lamination weight
Vlams = la.*Dso.^2*pi/4;
Qlams = Vlams*DensityFe*klam;
%% weight to be recycled
% stator slot
As0=(Dso.^2-Dsi.^2)*pi/4;
As=As0-((Hs+Htt).*Bs)*NoSlot;
Vs=la.*As*klam;
Msfe=Vs*DensityFe;
%rotor centre
Ar0=(Dro.^2-Dri^2)*pi/4;
Ar=Ar0-Wm.*Hm*2*2*p;        % 2 pics per pole for p pole pairs
Vr=la.*Ar*klam;   	%la.*Ar*(0.97-ksk);
Mrfe=Vr*DensityFe;
%total lam weight
QFe=Msfe+Mrfe;
Qlamnet=Qlams-QFe;


%% coil weight
pitch=2;    %shortpitch
DensityCu=8900; % Copper density, kg/m3
% Empirical parameters for coil end length calculation  
LcoilEndStraightPart=0.005;     %[m] Length of straight section of half coil end, single-side
DcoilEnd=0.003;                  % Diameter of coil end ring
AngleCoilEnd=58*pi/180;         % Angle between coil end and stack end section
NoPolePitch=NoSlot/(2*p)-pitch;     % Pole pitch - short pitched by x
lwend=1/cos(AngleCoilEnd)*NoPolePitch*pi*(Dsi+Hs)/NoSlot+2*LcoilEndStraightPart+pi*(DcoilEnd+Hs/2)/2; % length of single side end coil
Qcoil=DensityCu.*Bs.*Hs.*(la+lwend)*NoSlot*ksf;  % Total coil mass

%% magnet weight
Densitymag=7500;
Qmag=Wm.*Hm*2*2*p*Densitymag.*la;