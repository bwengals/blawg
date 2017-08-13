Title: Inducing point methods to speed up GPs
Date: 6/1/2017
Category: posts
Tags: gp, gsoc, gp-approximations
Another main avenue for speeding up GPs is inducing point methods, or sparse GPs.

<!-- PELICAN_END_SUMMARY -->



The idea is to reduce the effective number of input data points $x$ to the GP from $n$ to $m$, with $m < n$, where the set of $m$ points are called **inducing points**.  Since this makes the effective covariance matrix $K$ smaller, many inducing point approaches reduce the computational complexity from $\mathcal{O}(n^3)$ to $\mathcal{O}(nm^2)$.  The smaller $m$ is, the bigger the speed up.  

Many algorithms have been proposed to actually select the location of the $m$ inducing points.  The simplest approach is to select some subset of the $x$ points.  Doing this selection in a "smart" way is a difficult combinatorical optimization problem since the choices are discrete.  Another option is to choose $m$ points that aren't necessarily in the set of $x$ points but lie on the $x$ domain.  This is an easier, continuous, optimization problem.  The need to choose inducing points is common to all the GP approximation approaches.  Beyond the choice of inducing point locations, these methods then approximate either the GP prior, likelihood or both.  Sometimes this approximation is expressed as a modification of the GP prior covariance.  This post is for the most part a summary of [Quinonero-Candela & Rasmussen](http://www.jmlr.org/papers/v6/quinonero-candela05a.html) and [Wilson & Nickisch](https://arxiv.org/abs/1503.01057).


# 2.  Inducing point methods

The inducing points of the approximate GP can be viewed as a set of $m$ latent variables denoted $u$.  They are the values of the GP at the inducing input locations $z$.  The [GP prior distribution](https://bwengals.github.io/gp-speedups-1.html#Inferring-the-posterior-distribution-of-the-hyperparameters) (given hyperparameter values) is $p(f \mid \phi \,, \theta)$.  In our notation, $k(x, z)$ is a covariance **function** to be evaluated across $x$ and $z$, and $K_{x,z}$ is a covariance **matrix** that has been evaluated across $x$ and $z$.  The joint GP prior on $f$ and the prediction points $f_*$ is

$$
p(f, f_*) = N\left( 0 \,, \begin{bmatrix}
                            K_{x,x}   & K_{x, x_*} \\                            
                            K_{x_*, x} & K_{x_*, x_*} \\
                          \end{bmatrix} \right) \,.
$$

This is just the regular GP prior, where the new $x$ points, $x_*$ are included as a joint distribution.  We assume a zero mean function $\mu = 0$, and neglect $\phi$ and $\theta$ from the notation.  The inducing points are are another set of latent variables, so the GP prior can be recovered by marginalizing out the inducing points

$$
p(f, f_*) = \int p(f, f_*, u) \, du = \int p(f, f_* \mid u) p(u) \, du \,,
$$

where $p(u) = N(0\,, k(z, z'))$, where the covariance matrix obtained by evaluating $k(z, z')$ is $m \times m$.  [Quinonero-Candela & Rasmussen](http://www.jmlr.org/papers/v6/quinonero-candela05a.html) show that the majority of sparse GP approximations proposed in the literature approximate the GP prior $p(f, f_*)$ by assuming that **$f$ and $f_*$ are conditionally independent given the inducing points $u$**,

$$
p(f, f_*) \approx q(f, f_*) = \int q(f_* \mid u) q(f \mid u) p(u) \, du \,.
$$

- $q(f \mid u)$ is the **training conditional** distribution
- $q(f_* \mid u)$ is the **test conditional** distribution

It is also useful to define

$$
Q_{a,b} = K_{a, z} K_{z,z}^{-1} K_{z, b} \,.
$$

$f$ influences $f_*$ through $u$, which is why $u$ are called inducing variables.  $u$ is the value of the GP evaluated at $z$.

## SoR

The subset of regressors (SoR) approximation is the simplest, but it is not usually recommended for use in practice since its predictive uncertainties (not discussed here) are off the mark.  The approximations it proposes to use for the training and test conditional are

- $q(f, u) = N(K_{f,z} K_{z,z}^{-1} u\,, 0)$
- $q(f_*, u) = N(K_{*,z} K_{z,z}^{-1} u\,, 0)$

SoR imposes a deteriminstic relation between $f$ and $u$, since the covariance is zero.  SoR can also be viewed as having the effective covariance matrix

$$
\tilde{K}^{SoR} = K_{x,z} K_{z,z}^{-1} K_{z,x}  
$$

which is not of full rank ([KISSGP](https://arxiv.org/abs/1503.01057)).

## DTC

The deterministic training conditional (DTC) approximation doesn't suffer from the problems with the predictive uncertainties, but has the same predictive mean.  It has the same $p(f, u)$ as SoR,

- $q(f, u) = N(K_{f,z} K_{z,z}^{-1} u\,, 0)$
- $q(f_*, u) = p(f_*, u)$

and uses $p(f_*, u)$.

## FITC

Fully independent training conditional (FITC) is the sparse GP approach implemented in [GPML](http://www.gaussianprocess.org/gpml/code/matlab/doc/).

- $q(f \mid u) = \prod_{i=1}^{n} = N(K_{f,z} K_{z,z}^{-1}, diag[K_{f,f} - Q_{f,f}])$
- $q(f_* \mid u) = p(f_* \mid u)$

The relationship between $f$ and $u$ is not deterministic in this case.  The covariance is nonzero.  The effective covariance matrix for FITC is

$$
\tilde{K}^{FITC} = \tilde{K}^{SoR} + \delta_{xz}( K_{x,x'} - \tilde{K}^{SoR}) \,.
$$

It is full rank (DTC does not) due to diagonal correction to the covariance matrix.

## SKI (Structured Kernel Interpolation)

is described in the KISS-GP paper.  I'm most interested in this method right now, since it is a generalization of inducing point methods such as SoR and FITC.  It is also implemented in GPML. 

It approximates the true covariance matrix by *interpolating an $m \times m$ covariance matrix*,

$$
K_{x, z} \approx W K_{z,z} \,.
$$

Depending on the interpolation scheme used, $W$ can be very sparse.  For example, if $W$ represents local cubic interpolation, each row of $W$ will have 4 non-zero entries.  Since $W$ is sparse, matrix multiplications can use faster, sparse linear algebra routines.  Since we are interpolating $K_{z,z}$, we are free to choose inducing points that can [exploit matrix structure](https://bwengals.github.io/gp-speedups-2.html) such that $K_{u,u}$ has Toeplitz and/or Kronecker structure.  The true covariance matrix $K_{x,x}$ is approximated as

$$
K_{x,x} \approx W K_{z,z} W^T
$$

where $W$ is the matrix of interpolation weights.  The computational complexity of this approach depends on the number of inducing points used and how they are structured.  It isn't strictly necessary for the inducing points to be regularly spaced or on a multidimensional grid.  However, placing them on in a grid structure allows for a large number of inducing points to be used (even $m > n$), resulting in accurate interpolations.  The authors refer to SKI+Toeplitz+Kronecker as KISS-GP. It was recently implemented in the GPML version 4.0 MATLAB package.

### References

- [Quinonero-Candela & Rasmussen](http://www.jmlr.org/papers/v6/quinonero-candela05a.html)
- [KISS-GP](https://arxiv.org/abs/1503.01057)
- [GPML V4.0](http://www.gaussianprocess.org/gpml/code/matlab/doc/)
- [Wilson et al. 2015](https://arxiv.org/abs/1511.01870)


## Also

[GPflow](https://github.com/GPflow/GPflow) uses a sparse GP model distinct from those mentioned above.  It uses a variational approach to infer both the inducing points and the covariance function parameters.  I plan on researching their approach further.  See [here](http://gpflow.readthedocs.io/en/latest/notebooks/SGPR_notes.html) for a rundown. 

### Conclusion

Inducing point methods are attractive because they can largely be treated in a black box fashion.  However, as $m$ decreases, the quality of the approximation decreases.  A GP can be viewed as a model that fits a function using an infinite set of basis functions, which gives it more representation power than models with a fixed set of basis functions.  Unfortunately, using a smaller number of inducing points reduces the expressiveness of GPs.  One reason random forest type models or deep learning approaches have been successful on large datasets is their large number of parameters support expressive models.  It's a catch-22 that large data sets which would benefit the most from GP models necessitate approximations that degrade the representation power of GPs.  

At this point, the SKI approach where inducing points are placed on a grid, is the most promising approach to me.  
The GPML package implements most of the GP speed up methods discussed in the past few posts.  My next step will be to run timing tests using the GPML package to get a feel for the speedups and approximation quality offered by SKI vs FITC, and how much the Toeplitz or Kronecker structure in the inducing points contribute.
