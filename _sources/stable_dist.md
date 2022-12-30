# torchlevy.stable_dist

Supports the sample, pdf, and score functions from $\alpha$-stable distribution.

<!-- # Background

- $\alpha$-stable distribution
    
    The distribution for `stable_dist` has characteristic function:
    
    $$
    \varphi(t, \alpha, \beta, c, \mu)=e^{i t \mu-|c t|^\alpha(1-i \beta \operatorname{sign}(t) \Phi(\alpha, t))}
    $$
    
    where $*μ ∈ \R*$ is a shift parameter, $\beta \in [-1,1]$, called the *skewness parameter*
    , is a measure of asymmetry.
    
    where two different parameterizations are supported. 
    
    The first $S_1$:
    
    $$
    \Phi= \begin{cases}\tan \left(\frac{\pi \alpha}{2}\right) & \alpha \neq 1 \\ -\frac{2}{\pi} \log |t| & \alpha=1\end{cases}
    $$
    
    The second $S_0$:
    
    $$
    \Phi= \begin{cases}-\tan \left(\frac{\pi \alpha}{2}\right)\left(|c t|^{1-\alpha}-1\right) & \alpha \neq 1 \\ -\frac{2}{\pi} \log |c t| & \alpha=1\end{cases}
    $$
    
    In TorchLevy, the implementation follows second $S_1$.
    
    The probability density function for `stable_dist` is:
    
    $$
    f(x)=\frac{1}{2 \pi} \int_{-\infty}^{\infty} \varphi(t) e^{-i x t} d t
    $$
    
    where $−∞<t<∞$. This integral does not have a known closed form.
    
- isotropic $\alpha$-stable distribution
    
    1 dimmention에서는 symmetric 하면 isotropic이다. 하지만, n>2 부터는 symmteric 하면 isotropic이 되는 것은 아니다. multiple dimention에서 isotropic한 $\alpha$-stable distribution을 위한 equation은 다음과 같다. 
    
    $\mu=0, \beta=0$ 
    
    isotropic 하기 위해서는 mean이 0이 되고, skew가 없어야 합니다. 따라서 $*μ ∈ \R*$ is a shift parameter, $\beta \in [-1,1]$, called the *skewness paramete* , both become 0.
    
    Then the characteristic function과 pdf 가 단순화가 된다:
    
     $\varphi(t, \alpha, c)=e^{-|c t|^\alpha}$
    
    $f(x)=\frac{1}{2 \pi} \int_{-\infty}^{\infty} \varphi(t) e^{-i x t} d t= \frac{1}{2 \pi} \int_{-\infty}^{\infty} e^{-|c t|^\alpha} e^{-i x t} d t$
    
    이 함수를 multi dimention으로 확장하면 isotropic 하게 강제할 수 있다. 
    
    $f(\bold{x})=\frac{1}{(2 \pi)^d} \int_{\bold{u} \in \mathbb{R}^d} e^{-\|c\bold{u}\|^\alpha} e^{-i\langle \bold{x}, \bold{u}\rangle} d \bold{u}$ 
    
    - equation
        
        isotropic symmetric alpha stable
        
        $f(x) = \int_0^{\infty} e^{-r^\alpha} r^{\frac{d}{2}} J_{\frac{d}{2}-1}(r\|x\|) d r \frac{1}{\|x\|^{\frac{d}{2}-1}} \frac{1}{(2 \pi)^{\frac{d}{2}}}$, where $J$ is bessel function -->
        
    

# Function

## torchlevy.stable_dist.[sample](https://github.com/UNIST-LIM-Lab/torchlevy/blob/785de661b4e2819a8c2cb7af6884513c888fa14c/torchlevy/levy.py#L286-L304)

**Parameters:**

- `alpha` (float): The stability parameter of the distribution, must be in the range (0, 2].

- `beta` (float): The skewness parameter of the distribution, must be in the range [-1, 1].

- `size` (int or tuple of ints): The shape of the sample to generate.

- `loc` (float): The location parameter of the distribution.

- `scale` (float): The scale parameter of the distribution.

- `type` (torch.dtype): The data type of the sample.

- `reject_threshold` (float): The threshold for rejecting samples based on a criterion.

- `is_isotropic` (bool): Whether to generate an isotropic sample.

- `clamp_threshold` (float): The threshold for sample clamping

**Returns:**

- A sample from a symmetric alpha-stable distribution with the specified parameters

**Examples**:

```python
from torchlevy import stable_dist

sample = stable_dist.sample(alpha=1.5, size=10000)
print(f"min : {sample.min()}")
print(f"max : {sample.max()}")
# min : -544.177001953125
# max : 426.45806884765625
```

The code above serves as a prime illustration of the heavy tail characteristic inherent in the alpha stable distribution. It is evident that both the minimum and maximum values are considerably distant from the mode.


```python
import torch
from torchlevy import stable_dist
import matplotlib.pyplot as plt

sample = stable_dist.sample(alpha=1.7, size=10000)

plt.hist(sample.cpu(), 2000, facecolor='blue', alpha=0.5, label="samples")
plt.subplots_adjust(left=0.15)
plt.xlim(-15, 15)
plt.legend()
plt.show()
```

```{figure} assets/sample_dist.png
---
width: 30em
name: sample_dist
---
```

As demonstrated in the code above, the alpha stable sample conforms closely to the distribution of a ball shape.

```python
from torchlevy import stable_dist
import matplotlib.pyplot as plt

alpha = 1.5

plt.subplot(121)
non_isotropic_noise = stable_dist.sample(alpha, size=[10000, 2], is_isotropic=False).cpu()
plt.scatter(non_isotropic_noise[:, 0], non_isotropic_noise[:, 1], marker='.')
plt.gca().set_aspect('equal')
plt.xlim([-30, 30])
plt.ylim([-30, 30])
plt.title("non-isotropic")

plt.subplot(122)
isotropic_noise = stable_dist.sample(alpha, size=[10000, 2], is_isotropic=True).cpu()
plt.scatter(isotropic_noise[:, 0], isotropic_noise[:, 1], marker='.')
plt.gca().set_aspect('equal')
plt.xlim([-30, 30])
plt.ylim([-30, 30])
plt.title("isotropic")

plt.show()
```

![Untitled](assets/comparison.png)

The example above is a comparison between the **symmetric** alpha table distribution and the **isotropic** alpha table. In the plot on the left, the x and y coordinates of the point of dimension are independently sampled. As a result, you can see that most samples are near axis. On the other hand, in the plot on the right, the point was sampled from the 2d isotropc distribution. As a result, it can be seen that there is generally an invariant distribution in the direction.


---

## torchlevy.stable_dist.[pdf](https://github.com/UNIST-LIM-Lab/torchlevy/blob/785de661b4e2819a8c2cb7af6884513c888fa14c/torchlevy/levy.py#L13-L33)

**Parameters:**:

- `x`: a tensor of values at which the PDF is evaluated.

- `alpha`: alpha parameter of the symmetric alpha-stable distribution.

- `beta`: beta parameter of the symmetric alpha-stable distribution.

- `is_cache`: a boolean indicating whether sampling should be based on the linear interpolation of cached values within a 0.01 interval.

- `is_isotropic`: a boolean indicating whether the distribution should be isotropic, i.e. rotationally symmetrical and the same in all directions.

**Returns**:

- a tensor representing the PDF of the symmetric alpha-stable distribution.

**Examples**:

```python
import torch
from torchlevy import stable_dist

x = torch.arange(-2, 2, 0.5)
print(stable_dist.pdf(x, alpha=1.7))
# tensor([0.0928, 0.1477, 0.2108, 0.2633, 0.2840, 0.2633, 0.2108, 0.1477])
```

```python
x = torch.arange(-15, 15, 0.1)
alphas = [1.2, 1.5, 1.8]

for alpha in alphas:
    pdf = stable_dist.pdf(x, alpha)

    plt.plot(x.cpu(), pdf.cpu(), lw=2, label=f"alpha={alpha}")
    plt.xlim((-range_, range_))
    plt.ylim((0, 0.4))
    plt.legend()

plt.show()
```

```{figure} assets/dists.png
---
width: 30em
name: scores
---
```

---

## torchlevy.stable_dist.[score](https://github.com/UNIST-LIM-Lab/torchlevy/blob/785de661b4e2819a8c2cb7af6884513c888fa14c/torchlevy/levy.py#L182-L203)

**Parameters:**

- `x` : a tensor of values at which the score function is evaluated.

- `alpha` : a float representing the alpha parameter of the symmetric alpha-stable distribution.

- `beta` : a float representing the beta parameter of the symmetric alpha-stable distribution.

- `type` : a string indicating the type of score function to compute. Options are:

    - "cft": using the Fourier transform.

    - "cft2": using an alternative formulation of the Fourier transform.

    - "backpropagation": using backpropagation of the probability density function (PDF). Note: this option requires more memory.

- `is_isotropic` : a boolean indicating whether the distribution is isotropic.

- `is_fdsm` : a boolean indicating whether the score function is expressed as fractional DSM.

**Returns:**

- a tensor representing the score function of the symmetric alpha-stable distribution.

**Examples**:

```python
import torch
from torchlevy import stable_dist

x = torch.arange(-2, 2, 0.5)
print(stable_dist.score(x, alpha=1.7))
# tensor([ 1.1765,  0.8824,  0.5882,  0.2941,  0.0000, -0.2941, -0.5882, -0.8824])
```

```python
import torch
from torchlevy import stable_dist
import matplotlib.pyplot as plt

range_ = 5
x = torch.arange(-range_, range_, 0.1)

alphas = [1.2, 1.5, 1.8]

for alpha in alphas:
    score = stable_dist.score(x, alpha).cpu()

    plt.plot(x.cpu(), score, lw=2, label=f"alpha={alpha}")
    plt.xlim((-range_, range_))
    plt.legend()

plt.show()
```
```{figure} assets/scores.png
---
width: 30em
name: scores
---
```