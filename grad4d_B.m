% grad4d_B (script)
% 4D gradient descent/ascent experiment for 1-D analytic LS-GAN with exponential data Exp(c)
% Rayleigh latent variable z, square law generator xhat=gz^2+h
% discriminator D(x)=(1+erf(ax+b))/2
% The optimisation is over a, b, g & h.
% 
% c - data distribution parameter exponential(c) = 4.04 for case B
% a - discriminator parameter #1 = 1.21 for case B
% b - discriminator parameter #2 = -1 for case B
% g - generator parameter #1 = 0.35 for case B
% h - generator parameter #2 = -4 for case B

% Copyright 2020 - 2023 Graham Pulford
% Licence is granted to use, modify and distribute this code for non-commercial purposes provided that the original author's name 
% is referenced in any modified versions and in any supporting documentation.
% The following citation should be used for referencing this code:
% G. W. Pulford, Matlab code for "Convergence and Optimality Analysis of Low-Dimensional Generative Adversarial Networks Using Error Function Integrals",
% available from https://github.com/BandGapAI/gan-1d-least-squares.

debug=0; % set to 1 to print xmax

c=input('c (default 4.04 for Case B): ');
if isempty(c)
    c=4.04;
elseif c<=0
    error('c must be positive')
end
svec=[1;1;-1;-1];  % gradient ascent/descent

% Case B: pos neg peaks
a= 1.21;
b= -1;
g= 0.35;
h= -4;

NS=1e6;
theta0=[a b g h]';
theta=theta0;

K=input('Number of iterations K (default 250): ');
if isempty(K), K=250; end
step=input('Step size (default 0.4): ');
if isempty(step), step=0.4; end

Theta=zeros(K,4);
Theta(1,:)=theta;
J1=zeros(K,1);
J2=zeros(K,1);
[~,~,~,~,~,~,J1(1),J2(1)]=derivative1(a,b,c,g,h,NS);

disp(['   1: ',num2str(theta')])
for k=2:K
    [dJ1da,dJ1db,dJ2da,dJ2db,dJ2dg,dJ2dh,J1(k),J2(k)]=derivative1(theta(1),theta(2),c,theta(3),theta(4),NS,debug);
    Jgrad=[dJ1da+dJ2da;dJ1db+dJ2db;dJ2dg;dJ2dh];
    theta=theta+step*svec.*Jgrad;
    Theta(k,:)=theta;
    disp([sprintf('%4d',k),': ',num2str(theta')])
end

[~,~,~,~,~,~,J1ref,J2ref]=derivative1(a,b,c,1/c,0,NS);

KL=log(theta(3)*c)-1+(1-theta(4)*c)/(theta(3)*c);
disp(['KL divergence: ',num2str(KL)])

t=[1:K]';
figure(1); clf
plot(t,Theta(:,1),'k-',t,Theta(:,2),'b-',t,Theta(:,3),'g-',t,Theta(:,4),'r-')
xlabel('Iteration')
ylabel('Parameter Value')
title(['LS GAN GD 4D svec=[',num2str(svec'),']',' c=',num2str(c),' step=',num2str(step),' NS=',sprintf('%g',NS)])
legend(['a0=',num2str(a)],['b0=',num2str(b)],['g0=',num2str(g)],['h0=',num2str(h)])
grid

figure(2); clf
plot(t,J1,'b-',t,J2,'r',t,J1+J2,'k-')
xlabel('Iteration')
ylabel('Cost')
title(['LS GAN 4D Cost \theta_0=[',sprintf('%5g',theta0),'] c=',num2str(c),' step=',num2str(step),' NS=',sprintf('%g',NS)])
legend('J1','J2','J=J1+J2')
grid
