% grad4d_A (script)
% 4D gradient descent/ascent experiment for 1-D analytic LS-GAN with exponential data Exp(c)
% Rayleigh latent variable z, square law generator xhat=gz^2+h
% discriminator D(x)=(1+erf(ax+b))/2
% The optimisation is over a, b, g & h.
% 
% c - data distribution parameter exponential(c) = 0.5 for case A
% a - discriminator parameter #1 = 2.2 for case A
% b - discriminator parameter #2 = -3 for case A
% g - generator parameter #1 = 1.5 for case A
% h - generator parameter #2 = -2 for case A

% Copyright 2020 - 2025 Graham Pulford
% Licence is granted to use, modify and distribute this code for non-commercial purposes provided that the original author's name 
% is referenced in any modified versions and in any supporting documentation.
% The following citation should be used for referencing this code:
% G. W. Pulford, Matlab code for "Convergence and Optimality Analysis of Low-Dimensional Generative Adversarial Networks Using Error Function Integrals",
% available from https://github.com/BandGapAI/gan-1d-matlab.

debug=0; % set to 1 to print xmax

c=input('c (default 0.5 Case A): ');
if isempty(c)
    c=0.5;
elseif c<=0
    error('c must be positive')
end
svec=[1;1;-1;-1];  % gradient ascent/descent

% Case A
a= 2.2;
b= -3;
g= 1.5;
h= -2;

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
