# 1-D LS-GAN

This repository contains Matlab functions and scripts used to generate the quasi-analytical gradient ascent/descent results for the 1-D least-squares Generative Adversarial Network (1-D LS-GAN) in the following paper: [G. W. Pulford & K. Kondrashov, "Convergence and Optimality Analysis of Low-Dimensional Generative Adversarial Networks Using Error Function Integrals"](https://ieeexplore.ieee.org/document/9641798), pp. 165366 - 165384, Dec. 2021, IEEE Access. 

The 1-D LS-GAN is of the R/S/E/E type, which means that:
1. the latent variable z is Rayleigh distributed with PDF $p_Z(z)= 2z \exp(-z^2), z \ge 0$;
2. the generator is a square law $\hat x = G(z) = gz^2+h$  with real parameters $g>0, h$, resulting in an exponentially distributed generator output;
3. the discriminator is based on the error function $\text{erf}(x)$, specifically $D(x)=(1+\text{erf}(ax+b))/2$ with real parameters $a>0, b$;
4. the data are exponentially distributed with parameter $c>0: p_X(x)=c \exp(-cx), x \ge 0$.

Note that the Rayleigh & exponential PDFs used above are slighlty different from the standard definitions.

The code supplies the function value and gradients of the 1-D LS-GAN cost (loss) function for the quasi-analytical method based on known error function integrals that are computed via a key integral (integral8tilde_mc). The key integral is the integral from 0 to infinity of the function $\text{erf}(\alpha x+\beta) \cdot \text{Normal}(x,m,s)$ with respect to $x$, which must be evaluated numerically but is accurate to around 3 decimal places using 1-D Monte Carlo integration with 5E+06 samples. The MC integration is the only source of randomness in the analytical method, which is otherwise deterministic, hence the terminology "quasi-analytical".

Specifically, 

grad2d_A: generates top two plots of Fig 1 in paper;

grad4d_A: generates bottom two plots of Fig 1 in paper;

grad2d_B: generates top two plots of Fig 5 in paper;

grad4d_B: generates bottom two plots of Fig 5 in paper.

The scripts save the truth data to Jbh.mat, used to generate Figs 10 & 11 in the paper, however the surface plotting functions are not included.

Licence is granted to use, modify and distribute this code for non-commercial purposes provided that the original author's name is referenced in any modified versions and in any supporting documentation. The following citation should be used for referencing this code:

G. W. Pulford, Matlab code for "Convergence and Optimality Analysis of Low-Dimensional Generative Adversarial Networks Using Error Function Integrals", available from https://github.com/BandGapAI/gan-1d-matlab.

Copyright 2020 - 2023 Graham Pulford
