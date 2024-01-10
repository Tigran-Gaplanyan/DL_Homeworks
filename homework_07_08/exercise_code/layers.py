import numpy as np


       
def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.
    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the mean
    and variance of each feature, and these averages are used to normalize data
    at test-time.
    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:
    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var
    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7 implementation
    of batch normalization also uses running averages.
    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features
    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))
    out, cache = None, None
    if mode == 'train':
        sample_mean = np.mean(x, axis=0)

        x_minus_mean = x - sample_mean

        sq = x_minus_mean ** 2

        var = 1. / N * np.sum(sq, axis=0)

        sqrtvar = np.sqrt(var + eps)

        ivar = 1. / sqrtvar

        x_norm = x_minus_mean * ivar

        gammax = gamma * x_norm

        out = gammax + beta

        running_var = momentum * running_var + (1 - momentum) * var
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean

        cache = (out, x_norm, beta, gamma, x_minus_mean, ivar, sqrtvar, var, eps)

    elif mode == 'test':
        x = (x - running_mean) / np.sqrt(running_var)
        out = x * gamma + beta
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ########################################################################
    # TODO: Implement the backward pass for batch normalization.           #
    ########################################################################

    out, x_norm, beta, gamma, x_minus_mean, ivar, sqrtvar, var, eps = cache
    N, D = dout.shape

    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(x_norm * dout, axis=0)
    dx_norm = gamma * dout
    divar = np.sum(x_minus_mean * dx_norm, axis=0)
    dx_minus_mean1 = dx_norm * ivar
    dsqrtvar = -1. / (sqrtvar ** 2) * divar
    dvar = 0.5 * 1. / np.sqrt(var + eps) * dsqrtvar
    dsq = 1. / N * np.ones((N, D)) * dvar
    dx_minus_mean2 = 2 * x_minus_mean * dsq
    dx1 = (dx_minus_mean1 + dx_minus_mean2)
    dmu = -1 * np.sum(dx_minus_mean1 + dx_minus_mean2, axis=0)
    dx2 = 1. / N * np.ones((N, D)) * dmu
    dx = dx1 + dx2

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None

    ########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ########################################################################

    out, x_norm, beta, gamma, x_minus_mean, ivar, sqrtvar, var, eps = cache
    N, D = dout.shape

    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(x_norm * dout, axis=0)
    dx = (1. / N) * gamma * ivar * (N * dout - np.sum(dout, axis=0) - x_norm * np.sum(dout * x_norm, axis=0))


    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################

    return dx, dgamma, dbeta


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.
    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
        - mode: 'train' or 'test'; required
        - eps: Constant for numeric stability
        - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
        - running_mean: Array of shape (D,) giving running mean of features
        - running_var Array of shape (D,) giving running variance of features
    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.    #
    #                                                                      #
    # HINT: You can implement spatial batch normalization using the        #
    # vanilla version of batch normalization defined above. Your           #
    # implementation should be very short; ours is less than six lines.   #
    ########################################################################

    N, C, H, W = x.shape
    x_transposed = x.transpose(0, 2, 3, 1).reshape(-1, C)
    out_transposed, cache = batchnorm_forward(x_transposed, gamma, beta, bn_param)
    out = out_transposed.reshape(N, H, W, C).transpose(0, 3, 1, 2)

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.
    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass
    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.   #
    #                                                                      #
    # HINT: You can implement spatial batch normalization using the        #
    # vanilla version of batch normalization defined above. Your           #
    # implementation should be very short; ours is less than six lines.   #
    ########################################################################

    N, C, H, W = dout.shape
    dout_transposed = dout.transpose(0, 2, 3, 1).reshape(-1, C)
    dx_transposed, dgamma, dbeta = batchnorm_backward(dout_transposed, cache)
    dx = dx_transposed.reshape(N, H, W, C).transpose(0, 3, 1, 2)

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################

    return dx, dgamma, dbeta

def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not in
        real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        ########################################################################
        # TODO: Implement the training phase forward pass for inverted dropout. #
        # Store the dropout mask in the mask variable.                          #
        ########################################################################

        mask = (np.random.rand(*x.shape) < p) / p
        out = x * mask

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
    elif mode == 'test':
        ########################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.    #
        ########################################################################

        out = x

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        ########################################################################
        # TODO: Implement the training phase backward pass for inverted dropout. #
        ########################################################################

        dx = dout * mask

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
    elif mode == 'test':
        dx = dout
    return dx
