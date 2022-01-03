# Topic modelling
Analyze a document of words and figure out what the topic is. SPorts/

## Latent Dirichlet Allocation

continuous distribution on discrete probability vectors
Hearsay distribution

General process
- Generate each topic, which is a distribution on words. Bk should be sparse so topics have few relevant words. Words are learned.
- For each document, generate a distribution on topics
- For the nth word in the dth document 
  - allocate the word to a topic
  - generate the word from a topic



## Non negative matrix factorization

NMF


Minimizes one of the following:

constraint:  W and H contain non negative values

- Squared Error Objective 

- Divergence Objective


In this case maximum likilhood is equal to least squares


NMF vs SVD
- SVD has orthogonality constraint
- NMF has non-negativity constraint. It's interpretible 




## Principle Component Analysis

Consider if looking to reduce data down to one dimension, you would choose a unit vector that minimizes the sum of squared error projected on to the vector.

This vector is the first eigenvector.


PCA is looking for the k vectors that would minimize the same error. 



## Probablistic PCA
