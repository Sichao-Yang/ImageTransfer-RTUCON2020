function [lambda] = frict(Re,Rough)

%Colebrook equation approximations based on moody's diagram
Re_min=2000; 	% Re_min til 2000
Re_max=3000; 	% Re_max til 3000
for n=1:length(Rough)
if Rough(n)==0   
    Rough=1/1000/1000;
end
end
zeta_min=64/Re_min;

% Serghide's equation
A = -2*log10(Rough/3.7+12/Re_max);
B = -2*log10(Rough/3.7+2.51*A/Re_max);
C = -2*log10(Rough/3.7+2.51*B/Re_max);
zeta_max = (A-((B-A).^2./(C-2*B+A))).^(-2);

for n=1:length(Re)
if Re(n)<Re_min  lambda(n)=64/Re(n);
elseif Re(n)>Re_max  
    A = -2*log10(Rough/3.7+12./Re);
    B = -2*log10(Rough/3.7+2.51*A./Re);
    C = -2*log10(Rough/3.7+2.51*B./Re);
    lambda = (A-((B-A).^2./(C-2*B+A))).^(-2);
elseif and((Re(n)>=Re_min), (Re(n)<=Re_max))  
    lambda=zeta_min+(zeta_max-zeta_min)./(Re_max-Re_min)*(Re-Re_min);
end
end
% https://en.wikipedia.org/wiki/Darcy_friction_factor_formulae