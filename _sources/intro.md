# TorchLevy

![Untitled](assets/Untitled.png)

**Torchlevy** is a framework for PyTorch that enables the utilization of **$\alpha$-stable distributions** within the context of **Levy Processes**. It provides functionality for calculating probability density functions, generating samples, and calculating score associated with these distributions.
# Preliminary


```{admonition} Note$\qquad$$\alpha$-stable distribution
---
class: note, dropdown
---

```{figure} assets/Untitled1.png
---
width: 35em
name: ConvLNP
---
Symmetric $\alpha$-stable distributions at multiple $\alpha$. The tail gets heavier as $\alpha$ descreases

**Alpha-stable distributions** are a class of heavy-tailed probability distributions that are characterized by the fact that they are stable under linear combinations. Alpha-stable distributions are also known for their heavy-tailed behavior, which means that they have thicker tails than the normal distribution and are more likely to produce large deviations from the mean. 

**Symmetric** $\alpha$-stable distributions are a subclass of $\alpha$-stable distributions that are symmetric around the mean, which means that the distribution is the same whether you reflect it about the mean or not. In other words, if X is a random variable distributed according to a symmetric $\alpha$-stable distribution, then the distribution of -X is the same as the distribution of X.

**The heavy-tailed behavior** of $\alpha$-stable distributions is controlled by the parameter $\alpha \in (0, 2]$. When $\alpha < 2$, the distribution is said to have heavy tails, which means that it is more likely to produce large deviations from the mean. When $\alpha=2$ , the distribution is the same as the normal distribution and has lighter tails. 
```

    


```{admonition} Note$\qquad$Lévy process
---
class: note, dropdown
---

```{figure} assets/Untitled2.png
$\alpha$-stalbe Lévy process at multiple $\alpha$. As $\alpha$ gets smaller, jumps become larger and more frequent.

**Lévy process** is a type of stochastic process that is characterized by the fact that it has stationary and independent increments. This means that if $X(t)$ is a Lévy process, then the difference between $X(t+s)$ and $X(t)$ is independent of the value of $X(t)$ and is distributed according to the same distribution for all values of $t$ and $s$. Lévy processes are often used to model random processes in which the increments are uncorrelated and the distribution of the increments does not depend on the starting point.

Symmetric $\alpha$-stable distribution can be used to model **the increments of Lévy process**. In this case, the Lévy process is said to be an **$\alpha$-stable Lévy process**. Alpha-stable Lévy processes are characterized by heavy-tailed behavior, which means that they are more likely to produce large deviations from the mean than a normal Lévy process. This makes them useful for modeling processes that exhibit large fluctuations or have the potential for rare events.

One important property of $\alpha$-stable Lévy processes is the **jump property**, which states that the Lévy process can be decomposed into a continuous component and a jump component. The continuous component is similar to a Gaussian process, which means that it is continuous and has a bell-shaped distribution. Whereas the jump component resembles Poisson process, which means that it consists of a series of independent, discrete jumps. The jump component is responsible for the heavy-tailed behavior of the Lévy process and is what gives it the ability to produce large deviations from the mean.
```

    

# Why use TorchLevy?

Previously, the `class levy_stable` of `scipy` provided pdf and sampling functions for $\alpha$-stable distribution. However, the article below explains the advantage of using `torchlevy` rather than `scipy`.

## Performance boost
By utilizing TorchLevy, it is possible to achieve a staggering **performance boost of over x1000** in comparison to using [scipy's levy_stable](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.levy_stable.html) function. This means that computationally demanding tasks, such as those involved in deep learning, can be efficiently carried out with minimal overhead, including sampling and calculating probability density functions. For a more detailed breakdown of these figures, please refer to the table provided below.

|  | sampling | score | likelihood |
| --- | --- | --- | --- |
| scipy | 9.055s | 58.837s | 14.714s |
| TorchLevy | 0.009s (**x1000** faster) | 0.026s (**x2000** faster) | 0.003s (**x4000** faster) |


The significant increase was made possible through the implementation of two forms of parallel computing. The first involved the parallelization of input samples through the use of matrix operations, which enabled concurrent processing without interdependent inputs. The second involved the parallelization of integration through the transformation of both the likelihood and score into Fourier transforms and the utilization of numerical integration techniques. In order to facilitate this parallelism, we employed the python module [torchquad](https://github.com/esa/torchquad).


## Isotropic $\alpha$-stable distribution

```{figure} assets/sample_plot_comparison.png
---
width: 30em
name: sample_plot_comparison
---
2D samples from symmetric  $\alpha$-stable distribution and isotropic $\alpha$-stable distribution. Symmtric distribution shows depedency of distribution and direction.
```


The extension of the $\alpha$-stable distribution to $n$ dimensions results in a distribution that **lacks isotropy**, unlike the n-dimensional normal distribution. We propose an **isotropic alpha-stable distribution**, which preserves the heavy tail characteristics of the standard $\alpha$-stable distribution while also exhibiting isotropy. TorchLevy library offers sampling, probability density function, and score function implementations for the isotropic $\alpha$-stable distribution. These functions can be easily accessed by setting the parameter `is_isotropic=True` when calling functions from `class LevyStable`.

<!-- ## Combined score of normal and $\alpha$-stable distribution

The Time-reversal formula for SDEs with Lévy Processes is given by the following equation. 

```{math}
d \overleftarrow{X}_t=\left(b\left(t, \overleftarrow{X}_t\right)-\sigma_B^2(t) \partial_x \log p_t\left(\overleftarrow{X}_t\right)-\alpha \cdot \sigma_L^\alpha(t) \frac{\partial_{|x|}^{\alpha-2} \nabla_x p_t\left(\vec{X}_t\right)}{p_t\left(\vec{X}_t\right)}\right) d t+\sigma_B(t) d \bar{B}_t+\sigma_L(t) d \bar{L}_t^\alpha
```

If $\sigma_B, \sigma_L>0$ in the above equation, then the combined score must be calculated. Fortunately, the TorchLevy library offers a convenient way to calculate this score through `class LevyGaussian` . -->

<!-- ## Rectified Enhanced Lévy Score (ReELS)

To denoise the large noise at the heavy tail without losing the nature of the Lévy score function, we propose Rectified Enhanced Lévy Score (ReELS) as follows:

```{math}
\operatorname{ReELS}_\alpha(x)=\left\{\begin{array}{ll}
S_\alpha(x) & : x \in I_\alpha \\
-\operatorname{sgn}(x) \hat{c}|x|^{\hat{\beta}} & : \text { otherwise }
\end{array} \quad \hat{\beta}(\alpha) \in(0,1) .\right.
``` -->


<!-- 
# Score-based generative model with Levy processes

Concluding our exploration of TorchLevy's capabilities, we present [score-based generative model with Levy processes](https://openreview.net/forum?id=ErzyBArv6Ue) which introduces Symmetric $\alpha$-stable noise as a replacement for traditional Gaussian noise in a diffusion model. This unique approach, made possible through the use of torchlevy, offers a novel approach to generating data. Our experimentation has demonstrated that this method performs comparably to DDPM, while achieving a faster convergence rate.

```{figure} assets/combined.gif
---
width: 35em
name: combined
---
comparison betweem Ours(left) and DDPM(right) sampling. Generation based on Levy process(left) shows faster reconstruction speed. 
```-->







