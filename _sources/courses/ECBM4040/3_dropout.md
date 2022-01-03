
# Methods For Improving Neural Networks


# Dropout
Deep neural networks are prone to overfitting. One method to combat this is dropout. Dropout is randomly ommiting nodes during the training process.


Implementation of dropout:

```
def dropout_forward(x, dropout_config, mode):
    """
    Dropout feedforward
    :param x: input tensor with shape (N, D)
    :param dropout_config: (dict)
                           enabled: (bool) indicate whether dropout is used.
                           keep_prob: (float) retention rate, usually range from 0.5 to 1.
    :param mode: (string) "train" or "test"
    :return:
    - out: a tensor with the same shape as x
    - cache: (train phase) cache a random dropout mask used in feedforward process
             (test phase) None
    """
    keep_prob = dropout_config.get("keep_prob", 0.7)
    gone_prob = 1 - keep_prob

    out, cache = None, None
    if mode == "train":
        cache = np.ones(x.shape)
        num_cols = len(cache[0])
        off_cols = random.sample(range(0, num_cols), int(gone_prob*num_cols)+1)
        cache[:, off_cols] = 0
        out = x * cache
    elif mode == "test":
        out = x
        cache = None
        
    return out, cache


def dropout_backward(dout, cache):
    """
    Dropout backward only for train phase.
    :param dout: a tensor with shape (N, D)
    :param cache: (tensor) mask, a tensor with the same shape as x
    :return: dx: the gradients transfering to the previous layer
    """

    dx = cache * dout

    return dx
```

# Batch Normalization
Reset the weights after each layer

## Forward

### During Training

### While running

## Backward