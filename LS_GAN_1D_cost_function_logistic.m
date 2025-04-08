% function J=LS_GAN_1D_cost_function_logistic

% Copyright 2020 - 2025 Graham Pulford
% Licence is granted to use, modify and distribute this code for non-commercial purposes provided that the original author's name 
% is referenced in any modified versions and in any supporting documentation.
% The following citation should be used for referencing this code:
% G. W. Pulford, Matlab code for "Quasi-Analytical Least-Squares Generative Adversarial Networks: Further 1-D Results and Extension to Two Data Dimensions",
% available from https://github.com/BandGapAI/gan-1d-matlab.

function [J,varargout]=LS_GAN_1D_cost_function_logistic(a,b,g,h,x,z)
debug=0;
if nargin~=6
     error('[J,J1,J2] = LS_GAN_1D_cost_function_logistic(a,b,g,h,x,z)')
end
if a<=0 || g<=0
    disp('LS_GAN_1D_cost_function_logistic: a, g must be positive')
    J=NaN;
    if nargout>1
        varargout{1}=NaN;
        varargout{2}=NaN;
    end
    return
end
if ~isvector(x) || ~isvector(z)
    error('LS_GAN_1D_cost_function_logistic: x & z must be vectors')
end

Nx=length(x);
Nz=length(z);

if Nx==0 || Nz==0
    error('LS_GAN_1D_cost_function_logistic: x & z must be non-empty vectors')
end

if Nx~=Nz
    error('LS_GAN_1D_cost_function_logistic: x & z must be vectors of same length')
end

NS=Nx; % N1, N2, Nz are equal

if debug
    disp(['a: ',num2str(a)])
    disp(['b: ',num2str(b)])
    disp(['g: ',num2str(g)])
    disp(['h: ',num2str(h)])
end

J1=0;
J2=0;
for i=1:NS
    J1_i=1/(1+a*exp(-b*x(i)))^2;
    J1=J1+J1_i;
    Gz=g*z(i)^2+h;
    aexpbGz=a*exp(-b*Gz);
    J2_i=(aexpbGz/(1+aexpbGz))^2;
    J2=J2+J2_i;
end
J1=J1/NS;
J2=J2/NS;
J=J1+J2;
if nargout>1
   varargout{1}=J1;
   varargout{2}=J2;
end
return
