%I5n = integral5n(n,a,b,c,d,[N],[do_plot],[integral8tilde])
% int_{0}^{infty} x^n erf(ax+b) exp(-c^2 x^2 + 2dx) dx, a>0, c>0, n>=0
% 
% Requires Tn_ab(), integral_I5_mc(), integral_I5()
% 
% Examples
% I5n = integral5n(3,1,1.5,2,0.5,1e6) % calls MC integral_I5()
% I5n = integral5n(3,1,1.5,2,0.5,1e6,1) % positive peak I5, with plot, calls integral_I5_mc()
% I5n = integral5n(3,1,-1.5,2,0.5,1e6,1) % negative peak I5, with plot, calls integral_I5_mc()
% I5n = integral5n(3,1,-1.5,2,0.5,1e6,0,-0.562) % bypass MC integral_I5() and supply its value

% Copyright 2020 - 2023 Graham Pulford
% Licence is granted to use, modify and distribute this code for non-commercial purposes provided that the original author's name 
% is referenced in any modified versions and in any supporting documentation.
% The following citation should be used for referencing this code:
% G. W. Pulford, Matlab code for "Convergence and Optimality Analysis of Low-Dimensional Generative AdversarialNetworks Using Error Function Integrals,"
% pp. 165366 - 165384, Dec. 2021, IEEE Access.

function I5n = integral5n(n,a,b,c,d,varargin)
if isnan(a) || isnan(b) || isnan(c) || isnan(d)
    error('NaN value found in input parameters!')
end
if isinf(a) || isinf(b) || isinf(c) || isinf(d)
    error('Inf value found in input parameters!')
end
if a<=0
    disp('a must be positive! set I5n=Inf')
    I5n=inf;
    return
end
if c<=0
    disp('c must be positive! set I5n=Inf')
    I5n=inf;
    return
end
if n<0
    error('n must be nonnegative!')
end

if nargin==5
    N=1e5; % number of random samples
    do_plot=0;
    integral8tilde=0;
elseif nargin==6
    N=varargin{1};
    do_plot=0;
    integral8tilde=0;
elseif nargin==7
    N=varargin{1};
    do_plot=varargin{2};
    integral8tilde=0;
elseif nargin==8
    N=varargin{1};
    do_plot=varargin{2};
    integral8tilde=varargin{3};
end

if isinf(integral8tilde)
    disp('integral8tilde is Inf! set I5n=Inf')
    I5n=inf;
    return
end

sqrtpi=sqrt(pi);
expb2=exp(-b^2);
c2=c^2;
sqrta2c2=sqrt(a^2+c2);
if n==0
    if do_plot
        I5n=integral_I5_mc(a,b,c,d,N,do_plot);
    else
        I5n=integral_I5(a,b,c,d,N,integral8tilde);
    end
elseif n==1
    I5n=0.5*erf(b)+d*integral5n(0,a,b,c,d,N,do_plot,integral8tilde)+(a*expb2/sqrtpi)*Tn_ab(0,sqrta2c2,d-a*b);
    I5n=I5n/c2;
else
    I5n=d*integral5n(n-1,a,b,c,d,N,do_plot,integral8tilde)+(3-n)/2*integral5n(n-2,a,b,c,d,N,do_plot,integral8tilde)+(a*expb2/sqrtpi)*Tn_ab(n-1,sqrta2c2,d-a*b);
    I5n=I5n/c2;
end
return
