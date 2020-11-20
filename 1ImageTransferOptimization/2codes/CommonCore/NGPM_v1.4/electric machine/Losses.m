%% core+magnet loss
kcorecor=2;     % correction factor for core loss, set as '1' in FE. so should be '1.5' here
Pfe=(Iron_loss+Mag_loss_pu)*kcorecor.*la;
%% copper loss
TemperatureCu=110;
ResistivityCu=1.7241e-08*(1+0.393/100*(TemperatureCu-20));  % Resistivity of copper
kac=1.0;        %loss factor considering ac loss, not considered here         
Pcu=ResistivityCu*(Ja/ksf).^2.*(Qcoil/DensityCu)*kac;

%% mech loss + stray loss
Pw=Power*0.005;
Pw=repmat(Pw,length(Pfe),1);

%% objective no.2 efficiency
y(:,2)=(Pw+Pfe+Pcu)/Power-1;