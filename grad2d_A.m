%Theta=grad2d_A(K,step,NS,c,a,b,g,h,no_plot)
% 2D version of gradient descent/ascent experiment for 1-D analytic LS-GAN with exponential data Exp(c)
% Rayleigh latent variable z, square law generator xhat=gz^2+h
% discriminator D(x)=(1+erf(ax+b))/2
% The optimisation is over b & h.
%
% Input
% K - Number of iterations (default 250)
% step - Step size (default 0.4)
% NS - number of samples for MC integration (default 1E6)
% c - data distribution parameter exponential(c) = 0.5 for case A
% a - discriminator parameter #1 = 2.2 for case A
% b - discriminator parameter #2 = -3 for case A
% g - generator parameter #1 = 1.5 for case A
% h - generator parameter #2 = -2 for case A
% no_plot - set to 0 to suppress plot
% 
% Output
% Theta - parameter vector (Kx4)
% 
% Examples
%Theta=grad2d_A; % use all default parameters for Case A
%Theta=grad2d_A(250,0.4,1E6,0.5,2.2,-3,1.5,-2); % full call with plot
%Theta=grad2d_A(250,0.4,1E6,0.5,2.2,-3,1.5,-2,1); % no plot enabled

% Copyright 2020 - 2025 Graham Pulford
% Licence is granted to use, modify and distribute this code for non-commercial purposes provided that the original author's name 
% is referenced in any modified versions and in any supporting documentation.
% The following citation should be used for referencing this code:
% G. W. Pulford, Matlab code for "Convergence and Optimality Analysis of Low-Dimensional Generative Adversarial Networks Using Error Function Integrals",
% available from https://github.com/BandGapAI/gan-1d-matlab.

function Theta=grad2d_A(varargin)
if nargin==0
    K=250;
    step=0.4;
    NS=1.0E+6;
    c=0.5;
    a= 2.2;
    b= -3;
    g= 1.5;
    h= -2;
    no_plot=0;
elseif nargin==1
    K=varargin{1};
    step=0.4;
    NS=1.0E+6;
    c=0.5;
    a= 2.2;
    b= -3;
    g= 1.5;
    h= -2;
    no_plot=0;
elseif nargin==2
    K=varargin{1};
    step=varargin{2};
    NS=1.0E+6;
    c=0.5;
    a= 2.2;
    b= -3;
    g= 1.5;
    h= -2;
    no_plot=0;
elseif nargin==3
    K=varargin{1};
    step=varargin{2};
    NS=varargin{3};
    c=0.5;
    a= 2.2;
    b= -3;
    g= 1.5;
    h= -2;
    no_plot=0;
elseif nargin==4
    K=varargin{1};
    step=varargin{2};
    NS=varargin{3};
    c=varargin{4};
    a= 2.2;
    b= -3;
    g= 1.5;
    h= -2;
    no_plot=0;
elseif nargin==5
    K=varargin{1};
    step=varargin{2};
    NS=varargin{3};
    c=varargin{4};
    a=varargin{5};
    b= -3;
    g= 1.5;
    h= -2;
    no_plot=0;
elseif nargin==6
    K=varargin{1};
    step=varargin{2};
    NS=varargin{3};
    c=varargin{4};
    a=varargin{5};
    b=varargin{6};
    g= 1.5;
    h= -2;
    no_plot=0;
elseif nargin==7
    K=varargin{1};
    step=varargin{2};
    NS=varargin{3};
    c=varargin{4};
    a=varargin{5};
    b=varargin{6};
    g=varargin{7};
    h= -2;
    no_plot=0;
elseif nargin==8
    K=varargin{1};
    step=varargin{2};
    NS=varargin{3};
    c=varargin{4};
    a=varargin{5};
    b=varargin{6};
    g=varargin{7};
    h=varargin{8};
    no_plot=0;
elseif nargin==9
    K=varargin{1};
    step=varargin{2};
    NS=varargin{3};
    c=varargin{4};
    a=varargin{5};
    b=varargin{6};
    g=varargin{7};
    h=varargin{8};
    no_plot=varargin{9};
else
    error('Usage: Theta=grad2d_A(K,step,NS,c,a,b,g,h,no_plot)')
end

svec=[1;-1]; % gradient ascent/descent

theta=[b h]'; % optimise over b & h

Theta=zeros(K,2);
J1=zeros(K,1);
J2=zeros(K,1);
[~,~,~,~,~,~,J1(1),J2(1)]=derivative1(a,b,c,g,h,NS);

Theta(1,:)=theta;
if no_plot==0
    disp(['   1: ',num2str(theta')])
end
for k=2:K
    [dJ1da,dJ1db,dJ2da,dJ2db,dJ2dg,dJ2dh,J1(k),J2(k)]=derivative1(a,theta(1),c,g,theta(2),NS);
    Jgrad2d=[dJ1db+dJ2db;dJ2dh];
    theta=theta+step*svec.*Jgrad2d;
    Theta(k,:)=theta;
    if no_plot==0
        disp([sprintf('%4d',k),': ',num2str(theta')])
    end
end

if no_plot==0
    ttxt=['LS GAN GD 2D s=[',num2str(svec'),'] a=',num2str(a),' c=',num2str(c),' g=',num2str(g),' step=',num2str(step),' NS=',sprintf('%g',NS)];
    ttxt2=['LS GAN 2D Cost a=',num2str(a),' c=',num2str(c),' g=',num2str(g),' step=',num2str(step),' NS=',sprintf('%g',NS)];
    
    t=[1:K]';
    figure(1); clf
    plot(t,Theta(:,1),'b-',t,Theta(:,2),'r-')
    xlabel('Iteration')
    ylabel('Parameter Value')
    title(ttxt)
    legend(['b0=',num2str(b)],['h0=',num2str(h)])
    grid
    
    figure(2); clf
    plot(t,J1,'b-',t,J2,'r',t,J1+J2,'k-')
    xlabel('Iteration')
    ylabel('Cost')
    title(ttxt2)
    legend('J1','J2','J=J1+J2')
    grid
end

% Theta has 2 columns, use a & g to form Kx4 Theta matrix
Theta=[a*ones(K,1) Theta(:,1) g*ones(K,1) Theta(:,2)];
return