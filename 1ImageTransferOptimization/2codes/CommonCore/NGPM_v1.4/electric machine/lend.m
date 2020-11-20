%% stator copper cost
QuoCu=68;       %[rmb/kg] Quotation of copper
DensityCu=8900; % Copper density, kg/m3
% Empirical parameters for coil end length calculation  
LcoilEndStraightPart=0.025;     %[m] Length of straight section of half coil end, single-side
DcoilEnd=0.03;                  % Diameter of coil end ring
AngleCoilEnd=58*pi/180;         % Angle between coil end and stack end section
p=4;
NoSlot=96; pitch=3;
NoPolePitch=NoSlot/(2*p)-pitch;     % Pole pitch - short pitched by x
Dsi=0.848;   Hs=0.063;
lwend=1/cos(AngleCoilEnd)*NoPolePitch*pi*(Dsi+Hs)/NoSlot+2*LcoilEndStraightPart+pi*(DcoilEnd+Hs/2)/2; % length of single side end coil
la=562.5/1000;
lw=la+lwend;   lw/la