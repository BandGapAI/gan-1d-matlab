%I9 = integral9(a,b,c)
% int_{0}^{infty} x erf(ax+b) c exp(-cx) dx, a>0, c>0
% 
% Requires Tn_ab()
% 
% Example
%I9 = integral9(2,-0.5,1)

% Copyright 2020 - 2023 Graham Pulford
% Licence is granted to use, modify and distribute this code for non-commercial purposes provided that the original author's name 
% is referenced in any modified versions and in any supporting documentation.
% The following citation should be used for referencing this code:
% G. W. Pulford, Matlab code for "Convergence and Optimality Analysis of Low-Dimensional Generative AdversarialNetworks Using Error Function Integrals,"
% pp. 165366 - 165384, Dec. 2021, IEEE Access.

function I9 = integral9(a,b,c)
if isnan(a) || isnan(b) || isnan(c)
    error('NaN value found in input parameters!')
end
if isinf(a) || isinf(b) || isinf(c)
    error('Inf value found in input parameters!')
end
if a<=0
    disp('a must be positive! set I9=Inf')
    I9=inf;
    return
end
if c<=0
    error('c must be positive!')
end
sqrtpi=sqrt(pi);
expb2=exp(-b^2);
abc2=a*b+c/2;
I9=erf(b)/c+(2*a*expb2/(c*sqrtpi))*Tn_ab(0,a,-abc2)+(2*a*expb2/sqrtpi)*Tn_ab(1,a,-abc2);
return
