%[I8hat,[xmax]]=integral8tilde_mc(alpha,beta,m,s,[N],[do_plot])
% 
% Integrate the function erf(alpha*x+beta) * Gaussian(x,m,s) in the region of interest using Monte Carlo integration.
% integral8tilde is the integral from 0 to infinity of this function.
% 
% The function call is vectorized for speed.
% 
% Caveat
% For certain cases the search fails to find a suitable value of xmax.
% This can be because of the location of the negative peak (if there is one) or the positive peak.
% 
% Examples
% I8hat=integral8tilde_mc(1,-1,0.3,0.5) % use default number of samples N
% I8hat=integral8tilde_mc(2,-1,0.3,0.5,1e6) % use N=1e6 samples
% I8hat=integral8tilde_mc(1,-2,0.3,0.5,1e6,1) % also do plot of function
% I8hat=integral8tilde_mc(1,-2,0.3,0.5,1e6,0,1) % enable debug
%
% Copyright 2020 - 2025 Graham Pulford
% Licence is granted to use, modify and distribute this code for non-commercial purposes provided that the original author's name 
% is referenced in any modified versions and in any supporting documentation.
% The following citation should be used for referencing this code:
% G. W. Pulford, Matlab code for "Convergence and Optimality Analysis of Low-Dimensional Generative Adversarial Networks Using Error Function Integrals",
% available from https://github.com/BandGapAI/gan-1d-matlab.

function [I8hat,varargout]=integral8tilde_mc(alpha,beta,m,s,varargin)
if isnan(alpha) || isnan(beta) || isnan(m) || isnan(s)
    error('NaN value found in input parameters!')
end
if isinf(alpha) || isinf(beta) || isinf(m) || isinf(s)
    error('Inf value found in input parameters!')
end
if nargin==4
    N=1e6; % number of random samples
    do_plot=0;
    debug=0;
elseif nargin==5
    N=varargin{1};
    do_plot=0;
    debug=0;
elseif nargin==6
    N=varargin{1};
    do_plot=varargin{2};
    debug=0;
elseif nargin==7
    N=varargin{1};
    do_plot=varargin{2};
    debug=varargin{3};
else
    error('[I8hat,[xmax]]=integral8tilde_mc(alpha,beta,m,s,[N],[do_plot],[debug])')
end
if alpha<=0
    disp('a must be positive! set I8hat=Inf')
    I8hat=inf;
    return
end
if s<=0
    error('s must be positive!')
end

EPS=1.0E-20;

% importance region 
xmin=0;
xmax=1000;

% Refine (reduce) xmax knowing that function eventually asymptotes to zero
ymin=EPS; % 1e-10;
y1=erf(alpha*xmax+beta)*gaussian_1d(xmax,m,s);
xmax_found=0;
if y1>ymin
    xmax_found=1;
    if debug
        disp(['xmax=',num2str(xmax),' too small!'])
    end
else % at start of loop y1<=ymin
    Npts=100; % 500
    pass=0;
    while ~xmax_found
        pass=pass+1;
        x1=linspace(xmin,xmax,Npts);
        y1=erf(alpha*x1+beta).*gaussian_1d(x1,m,s);
        for i=Npts-1:-1:1
            if y1(i)>ymin
                xmax=x1(i+1);
                xmax_found=1;
                break
            end
        end
        if xmax_found
            if debug
                disp(['xmax: ',num2str(xmax),' found at pass: ',num2str(pass)])
            end
            break
        else
            if debug
                disp(['pass: ',num2str(pass),' xmax: ',num2str(xmax)])
            end
            xmax=xmax/10;
            if xmax<EPS
                xmax=EPS;
                break
            end
        end
    end
end
if ~xmax_found % this happens when y1<ymin for all x1 in [xmin,xmax]
    if debug
        disp(['xmax not found! Defaulting to EPS=',num2str(EPS)])
    end
    xmax=EPS;
end
if do_plot
    Npts=200;
    % determine range for plot
    x1=linspace(xmin,xmax,Npts);
    y=erf(alpha*x1+beta).*gaussian_1d(x1,m,s);
    figure(1)
    plot(x1,y)
    title(['I_8~(a,b,m,s) = (2\pi s)^{-1/2} \int_{0}^{\infty} erf(ax+b) exp(-(x-m)^2/2s) dx;',' a=',num2str(alpha),' b=',num2str(beta) ' m=',num2str(m),' s=',num2str(s)],'Interpreter','tex')
    grid
    v1=[xmin;xmin];
    v2=[0;0.2];
    v3=[xmax,xmax];
    v4=[0;0.2];
    v5=[0,0];
    v6=[0,0.5];
    hold on
    plot(v1,v2,'k--')
    plot(v3,v4,'k--')
    plot(v5,v6,'k-')
    hold off
    text(xmin-0.6,0.1,'xmin')
    text(xmax+0.1,0.1,'xmax')
end

u=xmin*ones(1,N)+(xmax-xmin)*rand(1,N);
y=(xmax-xmin)*erf(alpha*u+beta*ones(1,N)).*gaussian_1d(u,m,s);
I8hat=mean(y);

if nargout==2
   varargout{1}=xmax; 
end
return

function g=gaussian_1d(x,m,s)
% vectorised version (x is vector, m & s scalar)
% m is the mean, s is the variance = sigma^2
sqrt2pis=sqrt(2*pi*s);
g=exp(-0.5*(x-m*ones(size(x))).^2./s)./sqrt2pis;
return
