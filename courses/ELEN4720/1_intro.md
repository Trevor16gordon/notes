## Introduction

## Data Modeling
Data modeling generally broken down into
```
 _____________
|   1.Data    |       ________________________      ______________________
 ‾‾‾‾‾‾‾‾‾‾‾‾‾    -> |3.Infer hidden variables| -> |4.Predict and explore |
 _____________        ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾      ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
|2.Build Model|
 ‾‾‾‾‾‾‾‾‾‾‾‾‾‾
 ```

## Example - Guassian Multivariate

\begin{align}
p\left( x|\mu ,\Sigma \right) :=\dfrac{1}{\left( 2\pi \right) ^{\dfrac{d}{2}}\sqrt{\det \left( \Sigma \right) }}\exp \left[ -\dfrac{1}{2}\left( x-\mu \right) ^{T}\Sigma ^{-1}\left( x-\mu \right) \right] 
\end{align}


**1.Data**
Data 
\begin{align} 
x_{1}\ldots x_{n}|x_{i}\in \mathbb{R} ^{d}
\end{align}

**2. Build Model**
In this case we choose a probabilistic model Gaussian. We know the distribution family but don't know the parameters for the model

\begin{align}
p\left( x|\theta \right) ,\theta =\left\{ \mu ,\Sigma \right\} 
\end{align}

We make an assumption about the data that it is i.i.d. (independant and identically distributed)
\begin{align}
\chi _{i}n^{iid}p\left( x|\theta \right) ,i=1,\ldots n
\end{align}

With the assumption that the data is iid when know that the joint probability distribution is equal to the product of the marginal distributions:

\begin{align}
p\left( x_{1},\ldots x_{n}|\theta \right) =\prod ^{n}_{i=1}p\left( x_{i}|\theta \right)
\end{align}

**3.Infer Hidden Variables**
Looking for the **maxiumum likelihood** we seek to find the value of \theta that maximizes the likelihood function

\begin{align}
\nabla _{\theta }\prod ^{n}_{i:=1}p\left( x_{i}|\theta \right) =0
\end{align}

**Logarithmic Trick**
Applying a log function does not change the location of the maximums of a function, just the value. If we apply the logarithm before deriving the gradient, it is easier to determine.

\begin{align}
\ln \left( \prod _{i}f_{i}\right) =\sum _{i}\ln \left( f_{1}\right) 
\end{align}