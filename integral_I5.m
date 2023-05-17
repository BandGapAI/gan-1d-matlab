%I5 = integral_I5(a,b,c,d,NS,[integral8tilde])
% int_{0}^{infty} erf(a*x+b) * exp(-c^2*x^2+2*d*x) dx, a>0, c>0
%
% Calls integral8tilde_mc() to integrate erf(alpha*x+beta)*Gaussian(x,m,s) using Monte Carlo integration with NS samples.
% Compare with integral_I5_mc(), which should give similar answer subject to MC integration variance
%
% Examples
%I5=integral_I5(2,-3,0.2,0.1,1e6)
%I5=integral_I5(3,-2.5,1.2,0.5,1e6)
%I5=integral_I5(3,-2.5,1.2,0.5,1e6,-0.278) % bypass MC integral and supply its value

% Copyright 2020 - 2023 Graham Pulford
% Licence is granted to use, modify and distribute this code for non-commercial purposes provided that the original author's name 
% is referenced in any modified versions and in any supporting documentation.
% The following citation should be used for referencing this code:
% G. W. Pulford, Matlab code for "Convergence and Optimality Analysis of Low-Dimensional Generative AdversarialNetworks Using Error Function Integrals,"
% pp. 165366 - 165384, Dec. 2021, IEEE Access.

function I5 = integral_I5(a,b,c,d,NS,varargin)
if isnan(a) || isnan(b) || isnan(c) || isnan(d)
    error('NaN value found in input parameters!')
end
if isinf(a) || isinf(b) || isinf(c) || isinf(d)
    error('Inf value found in input parameters!')
end
if nargin==5
    integral8tilde=0;
elseif nargin==6
    integral8tilde=varargin{1};
end
if isinf(integral8tilde)
    disp('integral8tilde is Inf! set I5=Inf')
    I5=inf;
    return
end
if a<=0
    disp('a must be positive! set I5=Inf')
    I5=inf;
    return
end
if c<=0
    disp('c must be positive! set I5=Inf')
    I5=inf;
    return
end
expd2c2=exp(d^2/c^2);
m=d/c^2;
s=1/(2*c^2);
if integral8tilde % bypass MC integral
    I5=(sqrt(pi)/c)*expd2c2*integral8tilde;
else % perform MC integral
    I5=(sqrt(pi)/c)*expd2c2*integral8tilde_mc(a,b,m,s,NS);
end
return
