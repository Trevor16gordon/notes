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