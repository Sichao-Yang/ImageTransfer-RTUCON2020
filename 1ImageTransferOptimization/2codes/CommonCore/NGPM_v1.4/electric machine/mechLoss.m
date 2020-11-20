% this program is made to calculate the mechanical loss of a inner rotor machine
%% friction loss
% so far, formula 1,2,3,4 all at similar level after tuning
% formula 3 is recommanded
function [Pw, Pb] = mechLoss(fe,la,ag,Rro,p)
z=0;
t_in=35;
t_eavg=70;
%Air properties
%http://en.wikipedia.org/wiki/Atmospheric_pressure
Laps   =    0.0065;    %[K/m] Temperature lapse
M      =    0.0289644; %[kg/mol] Molecular mass, dry air
g      =    9.80665;   %[m/s^2] Gravitational acceleration
R_u    =    8.31447;   %[J/(mol*K)] Universal gas constant
T_ref  =    288.15;    %[K] Reference temperature
Tc2k   =    273.15;    %[K] celsius to kelvin 
p_atm0 =    101325;    %[Pa] sea level standard atmospheric pressure
R_a    =    R_u / M;   %[J/(kg*K)] Specific gas constant for air
p_atm  =    p_atm0 * (1 - z*Laps/T_ref)^(g/(Laps*R_a));  %[Pa] Air Pressure
%air density
t_av   =    (t_in + t_eavg)/2;      %[oC] average air temperature
rho_a  =    p_atm/(R_a*(t_av + Tc2k));  %[kg/m3] average air density in the generator
%Dyn. viscosity: https://en.wikipedia.org/wiki/Viscosity
C      =   120;                     %[K] Sutherland's constant for the gaseous material. air
T0     =   291.15;                  %[K] reference temeprature
mu0    =   18.27e-6;                %[Pa.s] reference viscosity
lamda  =   mu0*(T0+C)/(T0^(3/2));   %[Pa.s.K^(-1/2)] constant for the gas
T      =   t_av+Tc2k;               %[K] actual temeprature
mu_a   =   lamda*T^(3/2)/(T+C);     %[Pa.s = kg/(m*s)] Dyn. viscosity
% %Kin. visocity https://en.wikipedia.org/wiki/Viscosity#Dynamic_.28shear.29_viscosity 
% nu_air =   mu_a/rho_a;              %[m2/s] Kin. viscosity

Dro=Rro*2;   %[m] rotor outer diameter
omega=fe/p*pi*2; %[rad/s] mechanical angular speed
Res=rho_a*omega*Dro*g/2/mu_a;

%% formula 3 [prediction of windage power loss in alternators NASA]
Rough1=1.0e-3;  %[mm] absolute roughness for lamination, typical value: %http://www.nuclear-power.net/nuclear-engineering/fluid-dynamics/major-head-loss-friction-loss/relative-roughness-of-pipe/
Rough=(Rough1./ag);   %relative roughness
Cd=frict(Res,Rough);
Pw=pi*Cd.*(Dro/2).^4*omega^3.*la;

%% --------------------------------------------------
%% bearing loss from [design of rotating...]
Db=0.2; Cb=0.005; F=2000*9.8;
Pb=0.5*omega*Cb*F*Db;
