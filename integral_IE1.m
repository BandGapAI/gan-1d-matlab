%I1 = integral_IE1(a,b,c)
% int_{0}^{infty} erf(ax+b) c exp(-cx) dx, a>0, c>0
% 
% Reference: A table of integrals of the Error functions - NIST, Ng & Geller 1968, Equation (A3)
% 
% Example
%I1 = integral_IE1(2,-0.5,1)

% Copyright 2020 - 2023 Graham Pulford
% Licence is granted to use, modify and distribute this code for non-commercial purposes provided that the original author's name 
% is referenced in any modified versions and in any supporting documentation.
% The following citation should be used for referencing this code:
% G. W. Pulford, Matlab code for "Convergence and Optimality Analysis of Low-Dimensional Generative AdversarialNetworks Using Error Function Integrals,"
% pp. 165366 - 165384, Dec. 2021, IEEE Access.

function I1 = integral_IE1(a,b,c)
if isnan(a) || isnan(b) || isnan(c)
    error('NaN value found in input parameters!')
end
if isinf(a) || isinf(b) || isinf(c)
    error('Inf value found in input parameters!')
end
if a<=0
    disp('a must be positive! set I1=Inf')
    I1=inf;
    return
end
if c<=0
    error('c must be positive!')
end
c2a=c/(2*a);
e1=c2a^2+b*c/a;
I1=erf(b)+exp(e1)*(1-erf(b+c2a));
return
