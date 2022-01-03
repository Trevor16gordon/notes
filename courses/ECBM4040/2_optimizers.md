
# Optimizers
In machine learning learning it is useful to be able to find optimal weights/bias/parameters for a selected model. In some cases there are closed form solutions to those problems. Often there is no closed form solution and an iterative approach is needed to train the model. The following are common optimizers used to achieve this.

## Gradient Descent
The gradient shows the direction of the highest increase/decrease. This vector is useful as it points a path towards the local minimum. By moving model parameters in the negative direction of the gradient (with respect to the cost function) the locally optimal values for the parameters can be found.

## Stochastic Gradient Descent (SGD)
Stochastic gradient operates by calculating the gradient on a subset of data then moving. It's disadvantages is that it can get stuck in local minimum. Also, stochastic gradient descent may have problems where the slope is much steeper in one direction than another but the direction with steep slope is actually a ravine. In this case SGD will oscillate along the steep direction while only moving slowly in the other direction.


## SGD With Momentum
SGD with momentum is an improvement on SGD as it's momentum dampens the oscillations and it can move towards the minimum more directly

## AdaGrad
Rather than using a constant learning rate, Adagrad uses an adaptive learning rate that is intended to give stronger weights to the dimensions that haven't been updated much in the past. The benefit of this method is that the learning rate does not need to be set. It's disadvantage is that the learning rate will become extremely small over time and updates will stop.

## Adam
Adaptive Moment Estimation. This method is similar to adagrad in that it has an adaptive learning rate for each dimension. This method keeps track of the running velocity and momentum for each dimension. The momentum part accounts for the exponentially decaying nature of the average of these gradients over time. The update rule of Adam is a combination of momentum and the RMSProp optimizer.

## Nadam
Nadam is a mix of using Adam and Nesterov accelerated gradient (NAG). Nadam modifies Adam by updating the momentum and using that to get a more accurate gradient step.

## Newtons Method
Newtons method using the second gradient to get a more direct path to the minima.

