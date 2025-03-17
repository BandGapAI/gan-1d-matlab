%I10 = integral10(a,b,c,[N],[do_plot],[integral8tilde])
% int_{0}^{infty} x erf^2(ax+b) c exp(-cx) dx, a>0, c>0
% 
% Requires integral5n
% 
% Example
%I10 = integral10(1,-1.5,2,1e6,1)

% Copyright 2020 - 2025 Graham Pulford
% Licence is granted to use, modify and distribute this code for non-commercial purposes provided that the original author's name 
% is referenced in any modified versions and in any supporting documentation.
% The following citation should be used for referencing this code:
% G. W. Pulford, Matlab code for "Convergence and Optimality Analysis of Low-Dimensional Generative Adversarial Networks Using Error Function Integrals",
% available from https://github.com/BandGapAI/gan-1d-matlab.

function I10 = integral10(a,b,c,varargin)
if isnan(a) || isnan(b) || isnan(c)
    error('NaN value found in input parameters!')
end
if isinf(a) || isinf(b) || isinf(c)
    error('Inf value found in input parameters!')
end
if a<=0
    disp('a must be positive! set I10=Inf')
    I10=inf;
    return
end
if c<=0
    error('c must be positive!')
end
if nargin==3
    N=1e5; % number of random samples
    do_plot=0;
    integral8tilde=0;
elseif nargin==4
    N=varargin{1};
    do_plot=0;
    integral8tilde=0;
elseif nargin==5
    N=varargin{1};
    do_plot=varargin{2};
    integral8tilde=0;
elseif nargin==6
    N=varargin{1};
    do_plot=varargin{2};
    integral8tilde=varargin{3};
end
if isinf(integral8tilde)
    disp('integral8tilde is Inf! set I10=Inf')
    I10=inf;
    return
end

sqrtpi=sqrt(pi);
expb2=exp(-b^2);
abc2=a*b+c/2;
I10=erf(b)^2/c+(4*a*expb2/(c*sqrtpi))*integral5n(0,a,b,a,-abc2,N,do_plot,integral8tilde)+(4*a*expb2/sqrtpi)*integral5n(1,a,b,a,-abc2,N,do_plot,integral8tilde);
return
