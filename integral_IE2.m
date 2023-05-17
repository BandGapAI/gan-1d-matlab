%I2 = integral_IE2(a,b,c,NS,[integral8tilde])
% int_{0}^{infty} erf^2(ax+b) c exp(-cx) dx, a>0, c>0
% NS number of random samples
%
% Requires integral_I5()
% 
% Example
%I2 = integral_IE2(2,-0.5,1,1e6)

% Copyright 2020 - 2023 Graham Pulford
% Licence is granted to use, modify and distribute this code for non-commercial purposes provided that the original author's name 
% is referenced in any modified versions and in any supporting documentation.
% The following citation should be used for referencing this code:
% G. W. Pulford, Matlab code for "Convergence and Optimality Analysis of Low-Dimensional Generative AdversarialNetworks Using Error Function Integrals,"
% pp. 165366 - 165384, Dec. 2021, IEEE Access.

function I2 = integral_IE2(a,b,c,NS,varargin)
if isnan(a) || isnan(b) || isnan(c)
    error('NaN value found in input parameters!')
end
if isinf(a) || isinf(b) || isinf(c)
    error('Inf value found in input parameters!')
end
if nargin==4
    integral8tilde=0;
elseif nargin==5
    integral8tilde=varargin{1};
end
if isinf(integral8tilde)
    disp('integral8tilde is Inf! set I2=Inf')
    I2=inf;
    return
end

if a<=0
    disp('a must be positive! set I2=Inf')
    I2=inf;
    return
end
if c<=0
    error('c must be positive!')
end
I2=erf(b)^2+4*a/sqrt(pi)*exp(-b^2)*integral_I5(a,b,a,-(c/2+a*b),NS,integral8tilde);
return
