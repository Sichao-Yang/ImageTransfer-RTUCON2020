% this is a simple script to plot the optimized objective results
clc;clear;close;
figure(1)
% load data
A=importdata("IPM_bestobj_2pp.mat");
B=importdata("IPM_bestobj_2pp_simp.mat");
x=[125:1:235];
y=interp1(B(:,1),B(:,2),x);
plot(A(:,1),A(:,2),'o')
hold on; plot(x,y,'-');
% mark the selected point
k=find(x==144);
plot(x(k),y(k),'r*');   
plot(144,0.025,'r+');
k=find(y==min(y));
plot(x(k),y(k),'rs'); hold off;
title('optimisation of the V shaped IPM - 2 pole pair - 50H70oC')
xlabel('loss/Pin')
ylabel('cost (kRMB)')