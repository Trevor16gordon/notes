#!/usr/bin/env python
# coding: utf-8

# # Neural Networks
# 
# ![Example](images/feed_forward_example.png)
# 
# # Important Papers
# \begin{align}
# \dfrac{2L}{\partial \widehat{y}}RSS=2\times \left( y-\widehat{y}\right) 
# \begin{aligned}\dfrac{2L}{\partial \widehat{y}}RSS=2\times \left( y-\widehat{y}\right) \\
# \dfrac{2\widehat{y}}{2h^{2}}signoid=\sigma \left( \widehat{y}\right) \left( 1-\sigma \left( \widehat{y}\right) \right) \\
# \dfrac{\partial h2}{\partial W2}=21\\
# \dfrac{2h^{2}}{\partial z^{1}}=W2\\
# \dfrac{221}{2h1}=f|^{1}=f|_{s:gmoid}^{1}=\sigma \left( z\right) )\left( 1-\sigma \left( zI\right) \right) \end{aligned}
# \end{align}
# ## Derive dL/dW2 and dL/dc2
# We want to update W1, W2, c1, c2.
# To do this we need to update using dL/dW1, dL/dW2, dL/dc1, dL/dc1
# These are calculated using the chain rule
# \begin{align}
#   \\{\frac{\partial L}{\partial yhat}} RSS 
#     &= 2*(y-yhat)
#   \\{\frac{\partial yhat}{\partial h2}} sigmoid 
#     &= \sigma (yhat) (1 - \sigma (yhat))
#   \\{\frac{\partial h2}{\partial W2}} 
#     &= z1
#   \\{\frac{\partial h2}{\partial C2}} 
#     &= 1
#   \\{\frac{\partial h2}{\partial z1}} 
#     &= W2
#   \\{\frac{\partial z1}{\partial h1}} 
#     &= f1' =f1'sigmoid = \sigma (z1) (1 - \sigma (z1))
#   \\{\frac{\partial h1}{\partial W1}} 
#     &= x1
#   \\{\frac{\partial h1}{\partial C1}} 
#     &= 1
# \end{align}

# # Example Neural Network From Scratch

# In[1]:


"""Trevor Gordon. 2021
    Implementation of a feed forward neural network from scratch.
    Drawing largely from https://dafriedman97.github.io/mlbook/content/c7/concept.html
    and https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi
    """

import numpy as np
import tqdm


class FeedForwardNeuralNetwork():
    """Feed Forward Neural Network
    This FFNN is fixed to have one hidden layer. It is defined by weights self.W1 and self.W2 and
    biases self.c1 and self.c2
    """

    def __init__(self) -> None:
        """Init function
        Class variables:
            self.n_hidden (int): Number of neurons in single hidden layer
            self.f1 (func): Activation function to apply to hidden layer output before next layer
                Only supporting sigmoid right now
        """
        self.n_hidden = 1
        self.f1 = self.sigmoid
        self.f1_prime = self.sigmoid_prime

    def sigmoid(self, x):
        """Sigmoid function to bound output between 0 and 1"""
        return 1 / (1 + np.exp(-1*x))

    def sigmoid_prime(self, x):
        """Derivative of the sigmoid function simplifies to this."""
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def initialize_params(self, len_input, len_hidden, len_output):
        """Initialize weights and biases randomly.
        Args:
            len_input (int): Number of neurons in X input layer
            len_hidden (int): Number of neurons in hidden layer
            len_output (int): Number of neurons in y output layer
        Returns:
            weights_1 (np.array): Rows for neurons in hidden layer and columns for 
                neurons in X input layer.
            weights_2 (np.array): Rows for neurons in y output layer and columns for neurons in
                hidden layer
            biases_1 (np.array): Rows for neurons in hidden layer and 1 column
            biases_2 (np.array): Rows for neurons in output layer and 1 column
        """
        weights_1 = np.random.randn(len_hidden, len_input)/5
        biases_1 = np.random.randn(len_hidden, 1)/5
        weights_2 = np.random.randn(len_output, len_hidden)/5
        biases_2 = np.random.randn(len_output, 1)/5
        return weights_1, biases_1, weights_2, biases_2

    def update_network_states(self):
        """Using weights, biases and activation function, calculate hidden layer and output
        Returns:
            h1 (np.array): Hidden layer with shape = (len_hidden, num_observations)
            z1 (np.array): Activated hidden layer with shape = (len_hidden, num_observations)
            h2 (np.array): Hidden layer 2 with shape = (len_output, num_observations)
            yhat (np.array): Output layer with shape = (len_output, num_observations)
        """
        h1 = np.dot(self.W1, self.X.T) + self.c1
        z1 = self.f1(h1)
        h2 = np.dot(self.W2, z1) + self.c2
        yhat = self.f1(h2)
        return h1, z1, h2, yhat

    def get_loss_gradient_one_observations(self,
                                           y_one_obsv,
                                           yhat_one_obsv,
                                           h2_one_obsv,
                                           h1_one_obsv,
                                           z1_one_obsv,
                                           x_input_one_obsv,
                                           len_hidden):
        """Given a single observation find the loss gradient.
        Derivations for all of these are shown in the images folder.
        Args:
            y_one_obsv (np.array): Single output. Shape = (len_output, 1)
            yhat_one_obsv (np.array): Single predicted output given current weights/biases
            h2_one_obsv (np.array): Current h2 hidden layer for single input x and weights/biases
            h1_one_obsv (np.array): Current h1 hidden layer for single input x and weights/biases
            z1_one_obsv (np.array): Current z1 value for single input x and weights/biases
            x_input_one_obsv (np.array): Single x input
            len_hidden (int): Number of neurons in hidden layer
        Returns:
            dL_dW1_one_obvs (np.array): Partial grad showing direction to move. Same shape as W1 
            dL_dc1_one_obvs (np.array): Partial grad showing direction to move. Same shape as c1 
            dL_dW2_one_obvs (np.array): Partial grad showing direction to move. Same shape as W2 
            dL_dc2_one_obvs (np.array): Partial grad showing direction to move. Same shape as c2
        """

        len_input = len(x_input_one_obsv)
        len_output = len(y_one_obsv)

        # Start at output by calculation error
        dL_dyhat = -2*(y_one_obsv - yhat_one_obsv).T  # (1, len_output)

        ## LAYER 2 ##
        # dyhat_dh2.shape = (len_output, len_output)
        dyhat_dh2 = np.diag(self.f1_prime(h2_one_obsv))

        # dh2_dc2.shape =  (len_output, len_output)
        # dh2_dc2 = 1 because bias is simply added
        dh2_dc2 = np.eye(len_output)

        # dh2_dW2.shape = (len_output, (len_output, len_hidden))
        dh2_dW2 = np.zeros((len_output, len_output, len_hidden))
        for i in range(len_output):
            dh2_dW2[i] = z1_one_obsv

        # dh2_dz1.shape = (len_output, len_hidden)
        dh2_dz1 = self.W2

        ## LAYER 1 ##
        # dz1_dh1.shape = (len_hidden, len_hidden)
        dz1_dh1 = np.diag(self.f1(h1_one_obsv)*(1-self.f1(h1_one_obsv)))

        # dh1_dc1.shape = (len_hidden, len_hidden)
        dh1_dc1 = np.eye(len_hidden)

        # dh1_dW1
        # (len_hidden, (len_hidden, D_X))
        dh1_dW1 = np.zeros((len_hidden, len_hidden, len_input))
        for i in range(len_hidden):
            dh1_dW1[i] = x_input_one_obsv

        ## DERIVATIVES W.R.T. LOSS ##
        dL_dh2 = dL_dyhat @ dyhat_dh2
        dL_dW2_one_obvs = dL_dh2 @ dh2_dW2
        dL_dc2_one_obvs = dL_dh2 @ dh2_dc2

        dL_dh1 = dL_dh2 @ dh2_dz1 @ dz1_dh1

        dL_dW1_one_obvs = dL_dh1 @ dh1_dW1
        dL_dc1_one_obvs = dL_dh1 @ dh1_dc1
        # dL_dc1_one_obvs = dL_dc1_one_obvs.reshape(len(dL_dc1_one_obvs), -1)
        # dL_dc2_one_obvs = dL_dc2_one_obvs.reshape(len(dL_dc2_one_obvs), -1)
        return dL_dW1_one_obvs, dL_dc1_one_obvs, dL_dW2_one_obvs, dL_dc2_one_obvs

    def get_loss_gradient_iter_over_observations(self, len_hidden):
        """Get the cumulative gradient while iterating over each observation at a time
        Args:
            len_hidden (int): Number of neurons in hidden layer
        Returns:
            dL_dW1 (np.array): Partial grad showing direction to move. Same shape as W1 
            dL_dc1 (np.array): Partial grad showing direction to move. Same shape as c1 
            dL_dW2 (np.array): Partial grad showing direction to move. Same shape as W2 
            dL_dc2 (np.array): Partial grad showing direction to move. Same shape as c2
        """
        dL_dW2 = 0
        dL_dc2 = 0
        dL_dW1 = 0
        dL_dc1 = 0
        num_observations = len(self.X)
        for n in range(num_observations):
            # Slice y and yhat by observation number
            # dL_dyhat
            y_one_obsv = self.y[n]  # (1, len_output)
            yhat_one_obsv = self.yhat[:, n]  # (1, len_output)
            h2_one_obsv = self.h2[:, n]
            h1_one_obsv = self.h1[:, n]
            z1_one_obsv = self.z1[:, n]
            x_input_one_obsv = self.X[n]
            (dL_dW1_one_obvs, dL_dc1_one_obvs, dL_dW2_one_obvs,
             dL_dc2_one_obvs) = self.get_loss_gradient_one_observations(
                y_one_obsv, yhat_one_obsv, h2_one_obsv, h1_one_obsv,
                z1_one_obsv, x_input_one_obsv, len_hidden)
            dL_dW2 += dL_dW2_one_obvs
            dL_dc2 += dL_dc2_one_obvs
            dL_dW1 += dL_dW1_one_obvs
            dL_dc1 += dL_dc1_one_obvs
        return dL_dW1, dL_dc1, dL_dW2, dL_dc2

    def get_loss_gradient(self, len_hidden):
        """Get the loss gradient.
        This defaults to get_loss_gradient_iter_over_observations right now. In a future version
        this should be implemented while iterating on mini batches to improve speed.
        Args:
            len_hidden (int): Number of neurons in hidden layer
        Returns:
            dL_dW1 (np.array): Partial grad showing direction to move. Same shape as W1 
            dL_dc1 (np.array): Partial grad showing direction to move. Same shape as c1 
            dL_dW2 (np.array): Partial grad showing direction to move. Same shape as W2 
            dL_dc2 (np.array): Partial grad showing direction to move. Same shape as c2
        """
        (dL_dW1,
        dL_dc1,
        dL_dW2,
        dL_dc2) = self.get_loss_gradient_iter_over_observations(len_hidden)
        return dL_dW1, dL_dc1, dL_dW2, dL_dc2

    def fit(self, X, y, len_hidden, grad_step=1e-5, n_iter=1000, seed=None):
        """Fit a feedforward neural network to the test set given.
        Args:
            X (np.array): Input data with columns for elements of a single input
            and rows for number of observations. X.shape = (num_observations, data_per_observation)
            y (np.array): Correct prediction for the input samples. y.shape = (num_observations, len_output)
            len_hidden (int): Number of neurons in input layer.
            grad_step (float, optional): Step for updating weights with gradient vectors. Defaults to 1e-5.
            n_iter (int, optional): Number of iterations. Defaults to 1e3.
        """
        self.X = X
        # Reshaping with -1 make sure theres a second dim of at least 1
        self.y = y.reshape(len(y), -1)
        len_input = X.shape[1]
        len_output = self.y.shape[1]
        self.W1, self.c1, self.W2, self.c2 = self.initialize_params(
            len_input, len_hidden, len_output)
        self.h1, self.z1, self.h2, self.yhat = self.update_network_states()
        print(f"Starting to fit for {n_iter} iterations")
        for i in tqdm.tqdm(range(n_iter)):
            # Adjust weights and biases in the direction of the negative gradient of the loss function

            dL_dW1, dL_dc1, dL_dW2, dL_dc2 = self.get_loss_gradient(len_hidden)

            self.W1 -= grad_step * dL_dW1
            self.W2 -= grad_step * dL_dW2
            self.c1 -= grad_step * dL_dc1.reshape(-1, 1)
            self.c2 -= grad_step * dL_dc2.reshape(-1, 1)

            self.h1, self.z1, self.h2, self.yhat = self.update_network_states()
        return True

    def predict(self, X_predict):
        """Predict output for given input after model has been fit.
        Args:
            X_predict (np.array): Input data with columns for elements of a single input. 
            and  rows for each prediction. X.shape = (num_to_predict, data_per_observation)
        Returns:
            np.array: Predictions. y.shape = (data_per_output, num_to_predict)
        """
        self.X = X_predict
        h1, z1, h2, yhat = self.update_network_states()
        return yhat

