% Stochastic Gradient Ascent Descent (SGAD) experiment for 1-D Least Squares GAN (nx=nz=1)
% with 1-D latent variable z~Rayleigh p(z)=2z exp(-z^2), square law generator xhat=G(z)=gz^2+h (parameters g>0, h)
% Logistic discriminator D(x)=(1+a exp(-bx))^(-1) (parameters a>0, b)
% exponential data with parameter c>0: p_X(x)=c exp(-cx), x>=0

% Copyright 2020 - 2025 Graham Pulford
% Licence is granted to use, modify and distribute this code for non-commercial purposes provided that the original author's name 
% is referenced in any modified versions and in any supporting documentation.
% The following citation should be used for referencing this code:
% G. W. Pulford, Matlab code for "Quasi-Analytical Least-Squares Generative Adversarial Networks: Further 1-D Results and Extension to Two Data Dimensions",
% available from https://github.com/BandGapAI/gan-1d-matlab.

LW2=2; % linewidth for parameter plots

testcase=input('test case (default=1): ');
if isempty(testcase), testcase=1; end

if testcase==1
    c=0.5;
    a=0.7;
    b=-1.5;
    g=1.9;
    h=1.1;
    K=150;
    step=0.1;
elseif testcase==2
    c=0.3;
    a=1.5;
    b=1;
    g=0.9;
    h=-1.5;
    K=285;
    step=0.1;
elseif testcase==3 % higly variable trajectories -> chaotic
    c=1.1;
    a=1.9;
    b=0.2;
    g=1.4;
    h=-1.0;
    step=0.1;
    K=185;
elseif testcase==4 % higly variable trajectories -> regularly chaotic
    c=1.1;
    a=1.9;
    b=0.2;
    g=1.4;
    h=-0.9;
    step=0.1;
    K=185;
end

% optimal generator parameters
gstar=1/c;
hstar=0;

% optimal discriminator parameters
astar=1;
bstar=0;

svec=[1;1;-1;-1]; % 4-D
theta0=[a b g h]';
theta=theta0;

NS=1e3; % number of samples for approximation of 2D LSGAN cost function
z=raylrnd(1/sqrt(2),NS,1);
x=exprnd(1/c,NS,1);

Theta=zeros(K,4); % 8-D
Theta(1,:)=theta;
J1=zeros(K,1);
J2=zeros(K,1);
[~,J1(1),J2(1)]=LS_GAN_1D_cost_function_logistic(a,b,g,h,x,z);

disp(['   1: ',num2str(theta')])
for k=2:K % 4-D
    Jgrad=ls_gan_1d_derivative_logistic(theta(1),theta(2),theta(3),theta(4),x,z);
    theta=theta+step*svec.*Jgrad;
    [~,J1(k),J2(k)]=LS_GAN_1D_cost_function_logistic(theta(1),theta(2),theta(3),theta(4),x,z);
    Theta(k,:)=theta;
    disp([sprintf('%8d',k),': ',num2str(theta')])
end

t=[1:K]';
figure(5); clf
plot(t,Theta(:,1),'k-',t,Theta(:,2),'b-',t,Theta(:,3),'g-',t,Theta(:,4),'r-','LineWidth',LW2)
hold on
plot(K+1,gstar,'go',K+1,hstar,'ro',K+1,astar,'ko',K+8,bstar,'bo','MarkerSize',10,'LineWidth',3)
hold off
legend(['a=',num2str(a)],['b=',num2str(b)],['g=',num2str(g)],['h=',num2str(h)]); % ,['g_1^*=',num2str(g1star)],['g_2^*=',num2str(g2star)])
xlabel('Iteration')
ylabel('Parameter Value')
title(['1D Logistic LSGAN c=',num2str(c),' \epsilon=',num2str(step),' s=[',num2str(svec'),']'])
grid

figure(6); clf
plot(t,J1,'b-',t,J2,'r',t,J1+J2,'k-','LineWidth',LW2)
xlabel('Iteration')
ylabel('Cost')
title(['1D Logistic LSGAN \theta_0=[',sprintf('%5g',theta0),'] c=',num2str(c),' step=',num2str(step)])
legend('J_1','J_2','J=J_1+J_2')
grid

save Theta1d_logistic_samp Theta a b g h c step K NS J1 J2
disp('results saved to Theta1d_logistic_samp')
