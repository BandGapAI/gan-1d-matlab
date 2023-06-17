%Iq = Intq(k,q)
% int_{0}^{q} u^k exp(-u^2) du

% Copyright 2020 - 2023 Graham Pulford
% Licence is granted to use, modify and distribute this code for non-commercial purposes provided that the original author's name 
% is referenced in any modified versions and in any supporting documentation.
% The following citation should be used for referencing this code:
% G. W. Pulford, Matlab code for "Convergence and Optimality Analysis of Low-Dimensional Generative Adversarial Networks Using Error Function Integrals",
% available from https://github.com/BandGapAI/gan-1d-least-squares.

function Iq = Intq(k,q)
if q<0
    error('q must be non-negative!')
end
if q==0
    Iq=0.0;
    return
end
k=round(k);
sqrtpi=sqrt(pi);
expq2=exp(-q^2);
if k<0
    error('k must be non-negative integer!')
elseif k==0
   Iq=sqrtpi/2*erf(q);
   return
end
% can assume k>=1
if mod(k,2)==0
    k_even=1;
    r=k/2;
else % k is odd
    k_even=0;
    r=(k+1)/2;
end

if k_even % even k=2r-1, r=1,2,...
    gammar12=gamma(r+1/2);
    if isinf(q)
        Iq=0.5*gammar12;
        return
    end
    t=0;
    for j=0:r-1
        t=t+gammar12/gamma(r+1/2-j)*(q^(2*(r-j)-1));
    end
    Iq=0.5*gammar12*erf(q)-0.5*t*expq2;
else % odd k=2r, r=1,2,...
    gammar=gamma(r);
    if isinf(q)
        Iq=0.5*gammar;
        return
    end
    t=0;
    for j=0:r-1
        t=t+gammar/gamma(r-j)*(q^(2*(r-j-1)));
    end
    Iq=0.5*gammar-0.5*t*expq2;
end
return
