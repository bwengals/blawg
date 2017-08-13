Title: PyMC3 FITC/VFE implementation notes
Date: 6/29/2017
Category: posts
Tags: gp, gsoc, gp-approximations
This post shows in detail how FITC and VFE is implemented in PyMC3.

<!-- PELICAN_END_SUMMARY -->



I followed [this](http://gpflow.readthedocs.io/en/latest/notebooks/SGPR_notes.html) pretty closely.  It can't be used exactly, since I made changes to incorporate FITC.  Hopefully this is useful for anyone (or me later when I find inevitable bugs) who needs to go over the PyMC3 implementation in detail. 

$$
\newcommand{tr}[1]{\mathrm{tr}#1}
\newcommand{diag}[1]{\mathrm{diag}#1}
\newcommand{Qff}{\mathbf{Q}_{ff}}
\newcommand{Lam}{\boldsymbol\Lambda}
\newcommand{Lami}{\boldsymbol\Lambda^{-1}}
\newcommand{sn}{\sigma^2_n}
\newcommand{Kff}{\mathbf{K}_{ff}}
\newcommand{Kuu}{\mathbf{K}_{uu}}
\newcommand{Kuui}{\mathbf{K}^{-1}_{uu}}
\newcommand{Kuf}{\mathbf{K}_{uf}}
\newcommand{Kfu}{\mathbf{K}_{fu}}
\newcommand{G}{\mathbf{G}}
\newcommand{A}{\mathbf{A}}
\newcommand{Ai}{\mathbf{A}^{-1}}
\newcommand{At}{\mathbf{A}^{\mathrm{T}}}
\newcommand{B}{\mathbf{B}}
\newcommand{Bi}{\mathbf{B}^{-1}}
\newcommand{C}{\mathbf{C}}
\newcommand{Ct}{\mathbf{C}^{\mathrm{T}}}
\newcommand{I}{\mathbf{I}}
\newcommand{W}{\mathbf{W}}
\newcommand{U}{\mathbf{U}}
\newcommand{V}{\mathbf{V}}
\newcommand{Vt}{\mathbf{V}^{\mathrm{T}}}
\newcommand{Wi}{\mathbf{W}^{-1}}
\newcommand{T}{\mathbf{T}}
\newcommand{y}{\mathbf{y}}
\newcommand{c}{\mathbf{c}}
\newcommand{ct}{\mathbf{c}^{\mathrm{T}}}
\newcommand{m}{\mathbf{m}}
\newcommand{r}{\mathbf{r}}
\newcommand{rt}{\mathbf{r}^{\mathrm{T}}}
\newcommand{rl}{\mathbf{r}_{\Lambda}}
\newcommand{yt}{\mathbf{y}^{\mathrm{T}}}
\newcommand{L}{\mathbf{L}}
\newcommand{Lb}{\mathbf{L}_{B}}
\newcommand{Li}{\mathbf{L}^{-1}}
\newcommand{Lti}{\mathbf{L}^{-\mathrm{T}}}
\newcommand{Lt}{\mathbf{L}^{\mathrm{T}}}
\newcommand{Lb}{\mathbf{L}_{B}}
\newcommand{Lbt}{\mathbf{L}_{B}^{\mathrm{T}}}
\newcommand{Lbti}{\mathbf{L}_{B}^{-\mathrm{T}}}
\newcommand{Lbi}{\mathbf{L}_{B}^{-1}}
$$

Although the VFE and FITC approximations are derived from different principles, they have a similar log marginal likelihood.

$$
-L = \frac{n}{2}\log(2\pi) + \frac{1}{2} \log \det(\Qff - \Lam)
   + \frac{1}{2}(\y - \m)^{\mathrm{T}}(\Qff + \Lam)^{-1}(\y - \m)
   + \frac{1}{2\sn} \tr(\T)
$$

For FITC
- $\Lam = diag[\Kff - \Qff] + \sn \I$
- $\T = 0$

For VFE:
- $\Lam = \sn \I$
- $\T = \Kff - \Qff$

Throughout, $\Qff = \Kfu \Kuui \Kuf$, the Nystrom approximation to the full covariance matrix $\Kff$.  Also notice that $\Lam$ is diagonal, so is easy to invert.  A flag is passed into the PyMC3 class that implements this, choosing which $\Lam$ and $\T$ to use.  So here is the derivation so that it is in terms of $\Lam$.  This is done so that solves are used instead of direct inverses (especially $n \times n$ ones!), and only $m \times m$ Cholesky decompositions are done.

## Identities

Woodbury identity:

$$
(\A + \C\B\Ct)^{-1} = \Ai - \Ai \C (\Bi + \Ct \Ai \C)^{-1} \Ct \Ai
$$

Matrix determinant lemma:

$$
\det(\A + \U\W\Vt) = \det(\Wi + \Vt\Ai\U)\det(\W)\det(\A)
$$

## Quadratic term

The covariance matrix in the quadratic term is $n \times n$, so we need to use the Woodbury matrix identity:

$$
\begin{aligned}
(\Lam + \Qff)^{-1} &= \Lami - \Lami\Kfu(\Kuu + \Kuf\Lami\Kfu)^{-1} \Kuf\Lami \\
\end{aligned}
$$

Next rotate the inverse in brackets using the Cholesky factor of $\Kuu$ ($\Kuu$ is $m \times m$, where $m$ is the number of inducing points, $m < n$).  So $\Kuu = \L\Lt$, and  $\Kuui = \Lti\Li$.  We get a more stable matrix to do solves on.  Multiply by an identity matrix made out of $\L$ on each side,

$$
\begin{aligned}
(\Lam + \Qff)^{-1} 
  &= \Lami - \Lami\Kfu\Lti\Lt(\Kuu + \Kuf\Lami\Kfu)^{-1}\L\Li\Kuf\Lami \\
  &= \Lami - \Lami\Kfu\Lti[\Li (\Kuu + \Kuf\Lami\Kfu)\Lti]^{-1}\Li\Kuf\Lami \\
  &= \Lami - \Lami\Kfu\Lti[\I + \Li\Kuf\Lami\Kfu\Lti]^{-1}\Li\Kuf\Lami \,.
\end{aligned}
$$

Next define $\A = \Li\Kuf$, $\At = \Kfu\Lti$.  $\A$ is $m \times n$.

$$
\begin{aligned}
(\Lam + \Qff)^{-1} 
  &= \Lami - \Lami\At[\I + \A\Lami\At]^{-1}\A\Lami \,.
\end{aligned}
$$

Next define $\r = \y - \m$, where $\m$ is the mean function evaluated at the input $x$ locations.

$$
\begin{aligned}
\rt(\Lam + \Qff)^{-1}\r 
  &= \rt\Lami\r - \rt\Lami\At[\I + \A\Lami\At]^{-1}\A\Lami\r \,.
\end{aligned}
$$

Then, set $\B = \I + \A\Lami\At$, which is $m \times m$, and use its Cholesky decomposition, $\B = \Lb\Lbt$.

$$
\begin{aligned}
\rt(\Lam + \Qff)^{-1}\r 
  &= \rt\Lami\r - \rt\Lami\At\Lbti\Lbi\A\Lami\r \,.
\end{aligned}
$$

Next set $\rl = \Lami\r$, and $\c = \Lbi\A\rl$,

$$
\begin{aligned}
\rt(\Lam + \Qff)^{-1}\r 
  &= \rt\rl - \ct\c \,.
\end{aligned}
$$

## Term with determinant
Using the determinant lemma,

$$
\begin{aligned}
\det(\Lam + \Qff) 
  &= \det(\Lam + \Kuf\Lami\Kfu)\det(\Kuui)\det(\Lam) \\
  &= \det(\L\Lt + \Kuf\Lami\Kfu)\det(\Lti\Li)\det(\Lam) \\
  &= \det(\L\Lt + \Kuf\Lami\Kfu)\det(\Lti)\det(\Li)\det(\Lam) \\
  &= \det(\I + \Li\Kuf\Lami\Kfu\Lti)\det(\Lam) \\
  &= \det(\I + \A\Lami\At)\det(\Lam)\\
  &= \det(\B)\det(\Lam)\\
\end{aligned}
$$


## Trace term

The trace term is present for VFE, for FITC it equals zero.  Note that $\tr(\A\B) = \tr(\B\A)$.

$$
\begin{aligned}
-\frac{1}{2\sn}\tr(\Kff - \Qff) 
  &= -\frac{1}{2\sn}\tr(\Kff) + \frac{1}{2\sn}\tr(\Qff) \\
  &= -\frac{1}{2\sn}\tr(\Kff) + \frac{1}{2\sn}\tr(\Kfu\Kuui\Kuf) \\
  &= -\frac{1}{2\sn}\tr(\Kff) + \frac{1}{2\sn}\tr(\At\A) \\
  &= -\frac{1}{2\sn}\tr(\Kff) + \frac{1}{2\sn}\tr(\A\At) \\
\end{aligned}
$$

Both $\tr(\Kff)$ and $\tr(\A\At)$ can be evaluated without computing the full matrices.

$$
\newcommand{ms}{\mathbf{m}_{*}}
\newcommand{Ksu}{\mathbf{K}_{*u}}
\newcommand{Kus}{\mathbf{K}_{u*}}
\newcommand{Kss}{\mathbf{K}_{**}}
$$

## Prediction

The expressions are derived differently, but result in the exact same formula.  This is the FITC result below.

$$
p(f_*) = \mathcal{N}(\mu_*\,,\, \Sigma_*)
$$

where

$$
\begin{aligned}
\mu_* &= \ms + \Ksu(\Kuu + \Kuf \Lami \Kfu)^{-1}\Lami(\y - \m) \\
\Sigma_* &= \Kss - \Ksu\Kuui\Kus + \Ksu(\Kuu + \Kuf\Lami\Kfu)^{-1}\Kus
\end{aligned}
$$

Like what we did with the quadratic term, we multiply by an identity matrix made from the Cholesky factor on $\Kuu$.

$$
\begin{aligned}
(\Kuu + \Kuf \Lami \Kfu)^{-1} 
  &= \Lti\Lt(\Kuu + \Kuf\Lami\Kfu)^{-1}\L\Li \\
  &= \Lti\Bi\Li \\
  &= \Lti\Lbti\Lbi\Li \\
\end{aligned}
$$

The substitutions are done the same way with $\A$, and we eventually get the result,

$$
\begin{aligned}
\mu_* &= \Ksu \Lti\Lbti\c \\
\Sigma_* &= \Kss - \Ksu\Li\Lbti\Lbt\Li\Kus
\end{aligned}
$$


## Summary

math
