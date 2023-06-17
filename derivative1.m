%[dJ1da,dJ1db,dJ2da,dJ2db,dJ2dg,dJ2dh,[J1],[J2]]=derivative1(a,b,c,g,h,NS,[debug])
%
% Compute the first derivatives (gradients) of a 1-D Least Squares GAN analytic model with latent variable z~Rayleigh,
% square law generator xhat=G(z)=gz^2+h (parameters g>0, h) (resulting in an exponentially distributed generator output)
% D(x)=(1+erf(ax+b))/2 discriminator (parameters a>0, b) and exponential data with parameter c>0: p_X(x)=c exp(-cx), x>=0
%
% The Cost function is decomposed as J(D,G)=J1(D)+J2(D,G)
% The overall gradients are:
% dJ/da=dJ1da+dJ2da
% dJ/db=dJ1db+dJ2db
% dJ/dg=dJ2dg
% dJ/dh=dJ2dh
%
% NS number of random samples for Monte Carlo integral
%
% Requires integral8_mc()
%
% Examples
%[dJ1da,dJ1db,dJ2da,dJ2db,dJ2dg,dJ2dh]=derivative1(1,2.8,2.5,1,4,1e6), Jgrad=[dJ1da+dJ2da;dJ1db+dJ2db;dJ2dg;dJ2dh]
%[dJ1da,dJ1db,dJ2da,dJ2db,dJ2dg,dJ2dh,J1,J2]=derivative1(1,-2.8,2.5,1,-4,1e6)
%[dJ1da,dJ1db,dJ2da,dJ2db,dJ2dg,dJ2dh,J1,J2]=derivative1(1,-2.8,2.5,1,-4,1e6,1) % with debug output

% Copyright 2020 - 2023 Graham Pulford
% Licence is granted to use, modify and distribute this code for non-commercial purposes provided that the original author's name 
% is referenced in any modified versions and in any supporting documentation.
% The following citation should be used for referencing this code:
% G. W. Pulford, Matlab code for "Convergence and Optimality Analysis of Low-Dimensional Generative Adversarial Networks Using Error Function Integrals",
% available from https://github.com/BandGapAI/gan-1d-least-squares.

function [dJ1da,dJ1db,dJ2da,dJ2db,dJ2dg,dJ2dh,varargout]=derivative1(a,b,c,g,h,NS,varargin)
if nargin==6
    debug=0;
elseif nargin==7
    debug=varargin{1};
else
    error('[dJ1da,dJ1db,dJ2da,dJ2db,dJ2dg,dJ2dh,[J1],[J2]]=derivative1(a,b,c,g,h,NS,[debug])')
end
if isnan(a) || isnan(b) || isnan(c) || isnan(g) || isnan(h)
    error('NaN value found in input parameters!')
end
if isinf(a) || isinf(b) || isinf(c) || isinf(g) || isinf(h)
    error('Inf value found in input parameters!')
end
if a<=0
    if debug
        disp('a must be positive! set derivatives=Inf')
    end
    dJ1da=inf;
    dJ1db=inf;
    dJ2da=inf;
    dJ2db=inf;
    dJ2dg=inf;
    dJ2dh=inf;
    if nargout>6
        varargout{1}=inf;
        varargout{2}=inf;
    end
    return
end
if g<=0
    if debug
        disp('g must be positive! set derivatives=Inf')
    end
    dJ1da=inf;
    dJ1db=inf;
    dJ2da=inf;
    dJ2db=inf;
    dJ2dg=inf;
    dJ2dh=inf;
    if nargout>6
        varargout{1}=inf;
        varargout{2}=inf;
    end
    return
end
if c<=0
    error('c must be positive!')
end
if NS<10
    error('NS must be large positive integer (min 10)!')
end

do_plot=0;
eta=a*h+b;
beta1=-(a*b+c/2);
gamma1=-(a*eta+1/(2*g));
sqrtpi=sqrt(pi);
a2=a^2;
on_g=1/g;
on_g2=on_g^2;

CJ1=c*exp(-b^2)/sqrtpi;
CJ2=-exp(-eta^2)/(g*sqrtpi);

m1=beta1/a2;
s1=1/(2*a^2);
m2=gamma1/a2;
s2=1/(2*a^2);

[I8_1,xmax1]=integral8tilde_mc(a,b,m1,s1,NS);
I50_1=integral5n(0,a,b,a,beta1,NS,do_plot,I8_1); % equivalent to integral_I5(a,b,a,beta1,NS,I8_1);
I51_1=integral5n(1,a,b,a,beta1,NS,do_plot,I8_1);

Psi0_1=Tn_ab(0,a,beta1)+I50_1; % Psi_0+
Psi1_1=Tn_ab(1,a,beta1)+I51_1; % Psi_1+

dJ1da=CJ1*Psi1_1;
dJ1db=CJ1*Psi0_1;

[I8_2,xmax2]=integral8tilde_mc(a,eta,m2,s2,NS);
I50_2=integral5n(0,a,eta,a,gamma1,NS,do_plot,I8_2);
I51_2=integral5n(1,a,eta,a,gamma1,NS,do_plot,I8_2);

IE1=integral_IE1(a,b,c);
IE2=integral_IE2(a,b,c,NS,I8_1);
J1=(1+2*IE1+IE2)/4;

if debug
    disp(['xmax1: ',num2str(xmax1),'  xmax2: ',num2str(xmax2)])
end

Psi0_2=Tn_ab(0,a,gamma1)-I50_2; % Psi_0-
Psi1_2=Tn_ab(1,a,gamma1)-I51_2; % Psi_1-

IE1=integral_IE1(a,eta,1/g);
IE2=integral_IE2(a,eta,1/g,NS,I8_2);
I9=integral9(a,eta,1/g);
I10=integral10(a,eta,1/g,NS,do_plot,I8_2);

J2=(1-2*IE1+IE2)/4;

if isinf(I8_1) || isinf(I8_2) || isinf(I50_1) || isinf(I51_1) || isinf(I50_2) || isinf(I51_2) || isinf(J1) || isinf(J2)...
        || isinf(Psi0_1) || isinf(Psi1_1) || isinf(Psi0_2) || isinf(Psi1_2)
    if debug
        disp('Inf detected in derivative1! set derivatives=Inf')
    end
    dJ1da=inf;
    dJ1db=inf;
    dJ2da=inf;
    dJ2db=inf;
    dJ2dg=inf;
    dJ2dh=inf;
    if nargout>6
        varargout{1}=inf;
        varargout{2}=inf;
    end
    return
end

if isnan(I8_1) || isnan(I8_2) || isnan(I50_1) || isnan(I51_1) || isnan(I50_2) || isnan(I51_2) || isnan(J1) || isnan(J2)...
        || isnan(Psi0_1) || isnan(Psi1_1) || isnan(Psi0_2) || isnan(Psi1_2)
    if debug
        disp('NaN detected in derivative1! set derivatives=Inf')
    end
    dJ1da=inf;
    dJ1db=inf;
    dJ2da=inf;
    dJ2db=inf;
    dJ2dg=inf;
    dJ2dh=inf;
    if nargout>6
        varargout{1}=inf;
        varargout{2}=inf;
    end
    return
end

dJ2da=CJ2*(h*Psi0_2+Psi1_2);
dJ2db=CJ2*Psi0_2;
dJ2dh=a*dJ2db;
dJ2dg=(2*on_g*IE1-on_g*IE2-2*on_g2*I9+on_g2*I10)/4;
if nargout>6
    varargout{1}=J1;
    varargout{2}=J2;
end
return