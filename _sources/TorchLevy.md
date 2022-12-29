# TorchLevy

![Untitled](TorchLevy%20b029a35bcb694634834ece09afe85607/Untitled.png)

Torchlevy is a framework for PyTorch that enables the utilization of Symmetric alpha stable distributions within the context of Levy Processes. It provides functionality for calculating probability density functions, generating samples, and calculating score associated with these distributions.

# Preliminary

- Symmetric alpha-stable distribution
    
    ![Untitled](TorchLevy%20b029a35bcb694634834ece09afe85607/Untitled%201.png)
    
    Alpha-stable distributions are a class of heavy-tailed probability distributions that are characterized by the fact that they are stable under linear combinations. Alpha-stable distributions are also known for their heavy-tailed behavior, which means that they have thicker tails than the normal distribution and are more likely to produce large deviations from the mean. 
    
    Symmetric alpha-stable distributions are a subclass of alpha-stable distributions that are symmetric around the mean, which means that the distribution is the same whether you reflect it about the mean or not. In other words, if X is a random variable distributed according to a symmetric alpha-stable distribution, then the distribution of -X is the same as the distribution of X.
    
    The heavy-tailed behavior of alpha-stable distributions is controlled by the parameter alpha, which can take on values between 0 and 2. When alpha is less than 2, the distribution is said to have heavy tails, which means that it is more likely to produce large deviations from the mean. When alpha is equal to 2, the distribution is the same as the normal distribution and has lighter tails.
    
    - more detail
        
        The probability density function of a symmetric alpha-stable distribution is given by:
        
        $f(x) = \left( \frac{1}{2 \beta B(1/\alpha, 1/\alpha)} \right) |x|^{-1 - \alpha} \exp{\left( - \left( \frac{|x|}{\beta} \right)^\alpha \right)}$
        
        where B is the Beta function.
        
        The symmetric alpha-stable distribution has several important properties. For example, it is self-similar, which means that if X is a random variable drawn from a symmetric alpha-stable distribution, then for any positive constant c, the random variable cX will also be drawn from a symmetric alpha-stable distribution with the same parameters. In addition, the symmetric alpha-stable distribution is stable under linear transformations, which means that if X1 and X2 are independent random variables drawn from a symmetric alpha-stable distribution, then the linear combination aX1 + bX2 will also be drawn from a symmetric alpha-stable distribution.
        
        The symmetric alpha-stable distribution has many applications in fields such as finance, physics, and engineering, where it is used to model the distribution of returns on financial assets, the distribution of particle velocities in gases, and the distribution of signal amplitudes in communication systems, among other things.
        
- Lévy process
    
    ![Untitled](TorchLevy%20b029a35bcb694634834ece09afe85607/Untitled%202.png)
    
    A Lévy process is a type of stochastic process that is characterized by the fact that it has stationary and independent increments. This means that if X(t) is a Lévy process, then the difference between X(t+s) and X(t) is independent of the value of X(t) and is distributed according to the same distribution for all values of t and s. Lévy processes are often used to model random processes in which the increments are uncorrelated and the distribution of the increments does not depend on the starting point.
    
    A symmetric alpha-stable distribution can be used to model the increments of a Lévy process. In this case, the Lévy process is said to be an alpha-stable Lévy process. Alpha-stable Lévy processes are characterized by heavy-tailed behavior, which means that they are more likely to produce large deviations from the mean than a normal Lévy process. This makes them useful for modeling processes that exhibit large fluctuations or have the potential for rare events.
    
    One important property of Lévy processes is the jump property, which states that the Lévy process can be decomposed into a continuous component and a jump component. The continuous component is similar to a Gaussian process, which means that it is continuous and has a bell-shaped distribution. Whereas the jump component resembles Poisson process, which means that it consists of a series of independent, discrete jumps. The jump component is responsible for the heavy-tailed behavior of the Lévy process and is what gives it the ability to produce large deviations from the mean.
    
    - more detail
        
        One important property of Lévy processes is the Lévy-Khintchine representation, which states that the characteristic function (the Fourier transform of the distribution) of a Lévy process can be written in the form:
        
        phi(u) = exp(iux - q(u))
        
        where x is the mean of the distribution and q(u) is the Lévy measure, which describes the distribution of the increments. For an alpha-stable Lévy process, the Lévy measure is given by:
        
        q(u) = (|u|^alpha)/(2^(alpha-1) * gamma(alpha))
        
        where alpha is the stability parameter and gamma is the gamma function. The Lévy measure determines the behavior of the Lévy process and is used to compute the moments and other statistical properties of the process.
        
    

# Why use torchlevy?

By utilizing torchlevy, it is possible to achieve a staggering performance boost of over x1000 in comparison to using scipy's levy_stable function. This means that computationally demanding tasks, such as those involved in deep learning, can be efficiently carried out with minimal overhead, including sampling and calculating probability density functions. For a more detailed breakdown of these figures, please refer to the table provided below.

|  | sampling | score | likelihood |
| --- | --- | --- | --- |
| scipy | 9.055s | 58.837s | 14.714s |
| torchlevy | 0.009s (x1000 faster) | 0.026s (x2000 faster) | 0.003s (x4000 faster) |

In addition to providing significantly enhanced performance, torchlevy also offers a range of features that are not available in scipy. These include the ability to calculate the score for alpha stable and levy+gaussian distributions, as well as approximate scores for these distributions. Furthermore, torchlevy introduces the concept of isotropic alpha stable distributions and provides comprehensive support for sampling, probability density function calculations, and score computation for these distributions.

Concluding our exploration of torchlevy's capabilities, we present [score-based generative model with Levy processes](https://openreview.net/forum?id=ErzyBArv6Ue) which introduces Symmetric alpha stable noise as a replacement for traditional Gaussian noise in a diffusion model. This unique approach, made possible through the use of torchlevy, offers a novel approach to generating data. Our experimentation has demonstrated that this method performs comparably to DDPM, while achieving a faster convergence rate.

![left : ours, right : DDPM](TorchLevy%20b029a35bcb694634834ece09afe85607/combined.gif)

left : ours, right : DDPM

### 

`LevyGaussian` 

$d \vec{X}_t=b\left(t, \vec{X}_t\right) d t+\sigma_B(t) d B_t+\sigma_L(t) d L_t^\alpha, \quad t \in[0,1]$

$d \overleftarrow{X}_t=\left(b\left(t, \overleftarrow{X}_t\right)-\sigma_B^2(t) \partial_x \log p_t\left(\overleftarrow{X}_t\right)-\alpha \cdot \sigma_L^\alpha(t) \frac{\partial_{|x|}^{\alpha-2} \nabla p_t\left(\vec{X}_t\right)}{p_t\left(\vec{X}_t\right)}\right) d t+\sigma_B(t) d \bar{B}_t+\sigma_L(t) d \bar{L}_t^\alpha$

reverse form은 brownian motion과 levy process의 combined process의 score를 구해야 한다. 이 score를 구하기 위해 LevyGaussian을 지원합니다. 

`get_approx_score` 

- why use approximate score?