# python_Factor_Analyzer

This project is to implement varimax and promax rotations in factor analysis.

# Algorithm

## Varimax rotation

Give $\Phi$, is loadings matrix.
R is rotation given by varimax rotation.
I define $\Lambda$ is loadings matrix given by varimax rotation.
$$\Lambda = \Phi R$$
Solve $R$ under condition to maximize variance of $\Lambda$

### Solving Method
I use gradient method.
$\Lambda$ s gradient is defined by
$$A = \Lambda ^3 - \frac{\gamma}{p} \Lambda (diag(\Lambda ^T \Lambda)) $$
(caution: $\Lambda ^3$ is components of $\Lambda$ to the power of 3.)

$\Lambda ^3$ increases each component (loading) of $\Lambda$
$diag(\Lambda ^T \Lambda)$ increases enhanced diagonal components of $\Lambda$.
$p$ is the number of variables. $\frac{1}{p}$ decreases components of $\Lambda diag(\Lambda ^T \Lambda)$.
$\gamma$ is an adjustment weight.

Next,I decompose  $\Phi ^{T}A$ into singular values
$$A = U\Sigma V$$
$U$ and $V$ show kernel of $A$, so $R = UV$ updated.

The calculation ends when $R$ converges.


## Promax rotation

Assume promax loadings is $\Lambda ^{\kappa}$. (Defaults to 4.)

I init rotation matrix $T$ that is identity matrix.

$$ T_{next} = \({\Lambda ^{\kappa}})^{T} \Lambda T diag(\frac{1}{\({\Lambda ^{\kappa}})^{T} \Lambda T})  $$

The calculation ends when $T_{next} - T$ converges.
This formula is to make correlation matrix $({\Lambda^{\kappa}})^{T} \Lambda$ and adjust scale of $T$

# Directory

```
.
├── README.md
├── sandbox
│   └── README.md
└── src
    └── rotator.py
```
