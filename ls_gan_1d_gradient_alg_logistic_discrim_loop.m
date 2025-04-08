% Stochastic Gradient Ascent Descent (SGAD) experiment for 1-D Least Squares GAN (nx=nz=1)
% with 1-D latent variable z~Rayleigh p(z)=2z exp(-z^2), square law generator xhat=G(z)=gz^2+h (parameters g>0, h)
% Logistic discriminator D(x)=(1+a exp(-bx))^(-1) (parameters a>0, b)
% exponential data with parameter c>0: p_X(x)=c exp(-cx), x>=0
% Run in loop each test cases to help detect higly variable trajectories & possibly chaotic behaviour

% Copyright 2020 - 2025 Graham Pulford
% Licence is granted to use, modify and distribute this code for non-commercial purposes provided that the original author's name 
% is referenced in any modified versions and in any supporting documentation.
% The following citation should be used for referencing this code:
% G. W. Pulford, Matlab code for "Quasi-Analytical Least-Squares Generative Adversarial Networks: Further 1-D Results and Extension to Two Data Dimensions",
% available from https://github.com/BandGapAI/gan-1d-matlab.

LW2=1; % linewidth for parameter plots

testcase=input('test case (default=4): ');
if isempty(testcase), testcase=4; end

Nruns=input('Nruns (default=100): ');
if isempty(Nruns), Nruns=100; end

NS=input('NS (default=1e5): '); % number of samples for approximation of 2D LSGAN cost function
if isempty(NS), NS=1e5; end

if testcase==1
    c=0.5;
    a=0.7;
    b=-1.5;
    g=1.9;
    h=1.1;
    K=180;
    step=0.1;
    xl1=[0 K*1.03];
    xl2=[0 K*1.01];
    yl1=[-2.3 2.1];
    yl2=[-0.02 1.0];
elseif testcase==2
    c=0.3;
    a=1.5;
    b=1;
    g=0.9;
    h=-1.5;
    K=285;
    step=0.1;
    xl1=[0 K*1.03];
    xl2=[0 K*1.01];
    yl1=[-1.55 3.5];
    yl2=[-0.02 1.3];
elseif testcase==3 % higly variable trajectories
    c=1.1;
    a=1.9;
    b=0.2;
    g=1.4;
    h=-1.0;
    step=0.1;
    K=185;
    xl1=[0 K*1.03];
    xl2=[0 K*1.01];
    yl1=[-1.1 2.4];
    yl2=[-0.02 0.8];
elseif testcase==4 % higly variable trajectories
    c=1.1;
    a=1.9;
    b=0.2;
    g=1.4;
    h=-0.9;
    step=0.1;
    K=500;
    xl1=[0 K*1.03];
    xl2=[0 K*1.01];
    yl1=[-2.5 3.7];
    yl2=[-0.02 1.25];
end

% optimal generator parameters
gstar=1/c;
hstar=0;

% optimal discriminator parameters
astar=1;
bstar=0;

svec=[1;1;-1;-1]; % 4-D
theta0=[a b g h]';

Theta=NaN*ones(K,4,Nruns); % 4-D
J1=NaN*ones(K,Nruns);
J2=NaN*ones(K,Nruns);

for n=1:Nruns
    disp(['RUN ',num2str(n)])
    theta=theta0;
    z=raylrnd(1/sqrt(2),NS,1);
    x=exprnd(1/c,NS,1);
    Theta(1,:,n)=theta0;
    [~,J1(1,n),J2(1,n)]=LS_GAN_1D_cost_function_logistic(a,b,g,h,x,z);
    
    % disp([sprintf('%8d',1),': ',num2str(theta')])
    for k=2:K % 4-D
        Jgrad=ls_gan_1d_derivative_logistic(theta(1),theta(2),theta(3),theta(4),x,z);
        if any(isnan(Jgrad))
            disp(['NaN detected in Jgrad k=',num2str(k)])
            break
        end
        theta=theta+step*svec.*Jgrad;
        [~,J1(k,n),J2(k,n)]=LS_GAN_1D_cost_function_logistic(theta(1),theta(2),theta(3),theta(4),x,z);
        if isnan(J1(k,n)) || isnan(J2(k,n))
            disp(['NaN detected in J at k=',num2str(k)])
            break
        end        
        Theta(k,:,n)=theta;
    end
    disp([sprintf('%8d',k),': ',num2str(theta')])
end

t=[1:K]';
figure(5); clf
plot(t,Theta(:,1,1),'k-',t,Theta(:,2,1),'b-',t,Theta(:,3,1),'g-',t,Theta(:,4,1),'r-','LineWidth',LW2)
hold on
plot(K+1,gstar,'go',K+1,hstar,'ro',K+1,astar,'ko',K+8,bstar,'bo','MarkerSize',10,'LineWidth',3)
hold off
xlabel('Iteration')
ylabel('Parameter Value')
title(['1D Logistic LSGAN c=',num2str(c),' \epsilon=',num2str(step),' s=[',num2str(svec'),']'])
grid
for n=2:Nruns
    figure(5)
    hold on
    plot(t,Theta(:,1,n),'k-',t,Theta(:,2,n),'b-',t,Theta(:,3,n),'g-',t,Theta(:,4,n),'r-','LineWidth',LW2)
end
xlim(xl1)
ylim(yl1)
legend(['a=',num2str(a)],['b=',num2str(b)],['g=',num2str(g)],['h=',num2str(h)]); % ,['g_1^*=',num2str(g1star)],['g_2^*=',num2str(g2star)])

figure(6); clf
plot(t,J1(:,1),'b-',t,J2(:,1),'r',t,J1(:,1)+J2(:,1),'k-','LineWidth',LW2)
xlabel('Iteration')
ylabel('Cost')
title(['1D Logistic LSGAN \theta_0=[',sprintf('%5g',theta0),'] c=',num2str(c),' step=',num2str(step)])
grid
for n=2:Nruns
    figure(6)
    hold on
    plot(t,J1(:,n),'b-',t,J2(:,n),'r',t,J1(:,n)+J2(:,n),'k-','LineWidth',LW2)
end
xlim(xl2)
ylim(yl2)
legend('J_1','J_2','J=J_1+J_2')

save Theta1d_logistic_samp Theta a b g h c step K NS J1 J2
disp('results saved to Theta1d_logistic_samp')
