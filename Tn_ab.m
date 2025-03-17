%t = Tn_ab(n,a,b)
% 
% Tn(a,b) = int_{0}^{infty} x^n exp(-a^2 x^2 + 2bx) dx, a>0, n>=0
%         = a^(-n-1) exp(b^2/a^2) sum_{k=0}^{n} nchoosek(n,k) (b/a)^(n-k) Ik
% where
% Ik=Intq(k,Inf)+sign(b_on_a)*Intq(k,abs(b_on_a)), r=n/2; n even
% Ik=Intq(k,Inf)-Intq(k,abs(b_on_a)), r=(n+1)/2; n odd.
% 
% special case if n=0
% 
% Calls Intq(k,q) =  int_{0}^{q} u^k exp(-u^2) du
% 
% Reference: A table of integrals of the Error functions - NIST, Ng & Geller 1968
%
% Example
%Tn_ab(1,1,1)

% Copyright 2020 - 2025 Graham Pulford
% Licence is granted to use, modify and distribute this code for non-commercial purposes provided that the original author's name 
% is referenced in any modified versions and in any supporting documentation.
% The following citation should be used for referencing this code:
% G. W. Pulford, Matlab code for "Convergence and Optimality Analysis of Low-Dimensional Generative Adversarial Networks Using Error Function Integrals",
% available from https://github.com/BandGapAI/gan-1d-matlab.

function t = Tn_ab(n,a,b)
if isnan(a) || isnan(b)
    error('NaN value found in input parameters!')
end
if isinf(a) || isinf(b)
    error('Inf value found in input parameters!')
end
sqrtpi=sqrt(pi);
n=round(n);
if n<0
    error('n must be non-negative integer!')
end
if a<=0
    disp('a must be positive! set t=Inf')
    t=inf;
    return
end

b_on_a=b/a;

if n==0 % special case
    t=sqrtpi/(2*a)*exp(b_on_a^2)*(1+erf(b_on_a)); % valid for b of any sign
   return
end

t=0;
for k=0:n
    if mod(k,2)==0 % EVEN k=2r, r=1,2,...
        Ik=Intq(k,Inf)+sign(b_on_a)*Intq(k,abs(b_on_a));
    else % ODD k=2r-1, r=1,2,...
        Ik=Intq(k,Inf)-Intq(k,abs(b_on_a));
    end
    t=t+nchoosek(n,k)*(b_on_a^(n-k))*Ik;
end
t=t*exp(b_on_a^2)/a^(n+1);
return
