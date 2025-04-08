% function gradJ = ls_gan_1d_derivative_logistic

% Copyright 2020 - 2025 Graham Pulford
% Licence is granted to use, modify and distribute this code for non-commercial purposes provided that the original author's name 
% is referenced in any modified versions and in any supporting documentation.
% The following citation should be used for referencing this code:
% G. W. Pulford, Matlab code for "Quasi-Analytical Least-Squares Generative Adversarial Networks: Further 1-D Results and Extension to Two Data Dimensions",
% available from https://github.com/BandGapAI/gan-1d-matlab.

function gradJ=ls_gan_1d_derivative_logistic(a,b,g,h,x,z)
if nargin~=6
     error('gradJ=ls_gan_1d_derivative_logistic(a,b,g,h,x,z)')
end
if a<=0 || g<=0
    disp('ls_gan_1d_derivative_logistic: a, g must be positive')
    gradJ=NaN*ones(4,1);
    return
end
if ~isvector(x) || ~isvector(z)
    error('ls_gan_1d_derivative_logistic: x & z must be vectors')
end

Nx=length(x);
Nz=length(z);

if Nx==0 || Nz==0
    error('ls_gan_1d_derivative_logistic: x & z must be non-empty vectors')
end

if Nx~=Nz
    error('ls_gan_1d_derivative_logistic: x & z must be vectors of same length')
end

delta=1.0E-6; % finite difference

dJda=LS_GAN_1D_cost_function_logistic(a+delta,b,g,h,x,z)-LS_GAN_1D_cost_function_logistic(a-delta,b,g,h,x,z);
dJdb=LS_GAN_1D_cost_function_logistic(a,b+delta,g,h,x,z)-LS_GAN_1D_cost_function_logistic(a,b-delta,g,h,x,z);
dJdg=LS_GAN_1D_cost_function_logistic(a,b,g+delta,h,x,z)-LS_GAN_1D_cost_function_logistic(a,b,g-delta,h,x,z);
dJdh=LS_GAN_1D_cost_function_logistic(a,b,g,h+delta,x,z)-LS_GAN_1D_cost_function_logistic(a,b,g,h-delta,x,z);

gradJ=[dJda;dJdb;dJdg;dJdh]/(2*delta);
return