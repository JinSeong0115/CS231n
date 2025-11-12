from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    x_reshaped = x.reshape(x.shape[0], -1)
    out = x_reshaped.dot(w) + b
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    x_reshaped = x.reshape(x.shape[0], -1)
    
    dx = (dout@w.T).reshape(x.shape)
    dw = x_reshaped.T@dout
    db = np.sum(dout, axis=0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    out = np.maximum(0, x)  # np.maximum(): element-wise comparison / np.max(): aggregation
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    dx = dout * (x>0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None

    ###########################################################################
    # TODO: Copy over your solution from A1.
    ###########################################################################
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dx = np.zeros_like(x)
    num_train = x.shape[0]

    # compute the probabilities in numerically stable way
    scores = x - np.max(x, axis=1, keepdims=True)
    p = np.exp(scores)
    p /= np.sum(p, axis=1, keepdims=True)  # normalize
    
    logp = np.log(np.maximum(p, 1e-12))
    correct_logp = logp[np.arange(num_train), y]  # (N,)
    loss = -np.sum(correct_logp)  # negative log probability is the loss
    loss = loss/num_train  # normalized

    # gradient 
    p[np.arange(num_train), y] -= 1
    dx = p/num_train
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

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
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        batch_mean = np.mean(x, axis=0)  # (1, D)
        batch_var = np.var(x, axis=0)  # (1, D)
        batch_std = np.sqrt(batch_var+eps)  # (1, D)

        # batch normalization
        x_normalized = (x-batch_mean)/batch_std
        out = gamma*x_normalized + beta

        # running mean/var for test
        running_mean = momentum*running_mean + (1-momentum)*batch_mean
        running_var = momentum*running_var + (1-momentum)*batch_var

        cache = (x, batch_mean, batch_var, gamma, bn_param)
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        running_std = np.sqrt(running_var + eps)
        x_normalized = (x-running_mean) / running_std

        out = gamma*x_normalized + beta
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """Backward pass for batch normalization.

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
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    x, batch_mean, batch_var, gamma, bn_param = cache
    eps = bn_param.get("eps", 1e-5)
    N, D = dout.shape
    
    batch_std = np.sqrt(batch_var+eps)
    x_normalized = (x-batch_mean)/batch_std

    # dbeta
    dbeta = np.sum(dout, axis=0)  # (D,)

    # dgamma
    dgamma = np.sum(dout * x_normalized, axis=0)  # (D,)

    # dx_norm
    dx_norm = dout * gamma  # shape (N, D)

    inv_std = 1.0 / batch_std

    sum_dx_norm = np.sum(dx_norm, axis=0)
    sum_dx_norm_from_mean = np.sum(dx_norm*x_normalized, axis=0)

    # dx
    dx = inv_std * (dx_norm - (1/N)*sum_dx_norm - (1/N)*x_normalized*sum_dx_norm_from_mean)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    See the jupyter notebook for more hints.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    x, batch_mean, batch_var, gamma, bn_param = cache
    eps = bn_param.get("eps", 1e-5)
    N, D = dout.shape
    
    batch_std = np.sqrt(batch_var+eps)
    x_normalized = (x-batch_mean)/batch_std

    # dbeta
    dbeta = np.sum(dout, axis=0)  # (D,)

    # dgamma
    dgamma = np.sum(dout * x_normalized, axis=0)  # (D,)

    # dx = dout x dx_norm x dx(x_norm/x)
    # dx(x_norm/x) = dx_from_x + dx_from_mean + dx_from_var
    # dx_norm
    dx_norm = dout * gamma  # shape (N, D)

    inv_std = 1.0 / batch_std

    sum_dx_norm = np.sum(dx_norm, axis=0)
    sum_dx_norm_from_mean = np.sum(dx_norm*x_normalized, axis=0)

    # dx
    dx = inv_std * (dx_norm - (1/N)*sum_dx_norm - (1/N)*x_normalized*sum_dx_norm_from_mean)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    layer_mean = np.mean(x, axis=1, keepdims=True)  # (N, 1)
    layer_var = np.var(x, axis=1, keepdims=True)  # (N, 1)

    # normalization
    layer_std = np.sqrt(layer_var + eps)  # (N, 1)
    x_normalized = (x - layer_mean) / layer_std  # (N, D)

    # scale & shift
    out = gamma * x_normalized + beta

    cache = (layer_std, x_normalized, gamma)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    layer_std, x_normalized, gamma = cache
    N, D = dout.shape

    # dbeta
    dbeta = np.sum(dout, axis=0)  # (D,)

    # dgamma
    dgamma = np.sum(dout * x_normalized, axis=0)  # (D,)

    # dx = dout x dx_norm x dx(x_norm/x)
    # dx(x_norm/x) = dx_from_x + dx_from_mean + dx_from_var
    # dx_norm
    dx_norm = dout * gamma  # shape (N, D)

    inv_std = 1.0 / layer_std

    sum_dx_norm = np.sum(dx_norm, axis=1, keepdims=True)  # (N, 1)
    sum_dx_norm_from_mean = np.sum(dx_norm * x_normalized, axis=1, keepdims=True)  # (N, 1)

    # dx
    dx = inv_std * (dx_norm - (1/D)*sum_dx_norm - (1/D)*x_normalized*sum_dx_norm_from_mean)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """Forward pass for inverted dropout.

    Note that this is different from the vanilla version of dropout.
    Here, p is the probability of keeping a neuron output, as opposed to
    the probability of dropping a neuron output.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        mask = (np.random.rand(*x.shape) < p)
        
        out = x * mask
        out /= p  # Scaling by dividing by p
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        out = x
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """Backward pass for inverted dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        dx = dout * mask/dropout_param["p"]
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == "test":
        dx = dout  # pass
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # parameter
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    S, P = conv_param['stride'], conv_param['pad']

    # output volume
    # H_out = (H-H'+2p)//s + 1  /  W_out = (W-W'+2p)//s + 1
    H_out = (H-HH+2*P)//S + 1
    W_out = (W-WW+2*P)//S + 1

    # output
    out = np.zeros((N, F, H_out, W_out), dtype=x.dtype)

    # zero padding (batch, channel, height, width)
    x_padded = np.pad(x, ((0,0), (0,0), (P,P), (P,P)), mode="constant", constant_values=0)

    # Convolution
    for n in range(N):
      for f in range(F):
        for i in range(H_out):
          h_start = i * S
          h_end = h_start + HH
          for j in range(W_out):
            w_start = j * S
            w_end = w_start + WW

            # patch
            patch = x_padded[n, :, h_start:h_end, w_start:w_end]

            # output
            out[n, f, i, j] = np.sum(patch*w[f]) + b[f]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache

def conv_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    x, w, b, conv_param = cache
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    _, _, H_out, W_out = dout.shape

    S = conv_param['stride']
    P = conv_param['pad']

    # Initialize gradient variables to zero
    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    # Apply padding to input x as in pure propagation.
    x_padded = np.pad(x, ((0,0), (0,0), (P,P), (P,P)), mode='constant', constant_values=0)
    dx_padded = np.zeros_like(x_padded)

    for n in range(N):
      for f in range(F):
        for i in range(H_out):
          h_start = i * S
          h_end = h_start + HH
          for j in range(W_out):
            # patch index
            w_start = j * S
            w_end = w_start + WW

            # upstream gradient
            dout_upstream = dout[n, f, i, j]

            # db
            db[f] += dout_upstream
            # dw
            patch = x_padded[n, :, h_start:h_end, w_start:w_end]
            dw[f] += patch * dout_upstream
            # dx
            dx_padded[n, :, h_start:h_end, w_start:w_end] += w[f] * dout_upstream

    dx = dx_padded[:, :, P:-P, P:-P]  # Remove padding
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    N, C, H, W = x.shape
    PH = pool_param['pool_height']
    PW = pool_param['pool_width']
    S = pool_param['stride']

    # output volume
    H_out = ((H-PH) // S) + 1
    W_out = ((W-PW) // S) + 1

    # Initialize output volume
    out = np.zeros((N, C, H_out, W_out), dtype=x.dtype)

    for n in range(N):
      for c in range(C):
        for i in range(H_out):
          for j in range(W_out):
            h_start = i * S
            h_end = h_start + PH
            w_start = j * S
            w_end = w_start + PW
            
            # if c->: => channel information disappeared (reduced to C->1)
            patch = x[n, c, h_start:h_end, w_start:w_end]
            
            out[n,c,i,j] = np.max(patch)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    x, pool_param = cache
    N, C, H, W = x.shape

    pool_H = pool_param['pool_height']
    pool_W = pool_param['pool_width']
    S = pool_param['stride']

    # output volume
    H_out = ((H-pool_H) // S) + 1
    W_out = ((W-pool_W) // S) + 1

    dx = np.zeros_like(x)

    for n in range(N):
      for c in range(C):
        for i in range(H_out):
          for j in range(W_out):
            h_start = i * S
            h_end = h_start + pool_H
            w_start = j * S
            w_end = w_start + pool_W

            patch = x[n, c, h_start:h_end, w_start:w_end]
            max_patch = np.max(patch)
            
            mask = (patch==max_patch)

            dout_max = dout[n, c, i, j]
            dx[n, c, h_start:h_end, w_start:w_end][mask] += dout_max
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """Computes the forward pass for spatial batch normalization.

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

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    N, C, H, W = x.shape

    x_transposed = x.transpose(0,2,3,1)  # (N,H,W,C)
    x_reshaped = x_transposed.reshape(-1, C)  # (N*H*W,C)

    out_reshaped, cache = batchnorm_forward(x_reshaped, gamma, beta, bn_param)
    
    out_transposed = out_reshaped.reshape(N, H, W, C)
    out = out_transposed.transpose(0, 3, 1, 2)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    N, C, H, W = dout.shape
    
    dout_transposed = dout.transpose(0, 2, 3, 1) # (N, H, W, C)
    dout_reshaped = dout_transposed.reshape(-1, C) # (N*H*W, C)

    dx_reshaped, dgamma, dbeta = batchnorm_backward(dout_reshaped, cache)

    dx_transposed = dx_reshaped.reshape(N, H, W, C)
    dx = dx_transposed.transpose(0, 3, 1, 2)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """Computes the forward pass for spatial group normalization.
    
    In contrast to layer normalization, group normalization splits each entry in the data into G
    contiguous pieces, which it then normalizes independently. Per-feature shifting and scaling
    are then applied to the data, in a manner identical to that of batch normalization and layer
    normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (1, C, 1, 1)
    - beta: Shift parameter, of shape (1, C, 1, 1)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################
    N, C, H, W = x.shape
    
    x_reshaped = x.reshape(N, G, C // G, H, W)
    x_flatten = x_reshaped.reshape(N * G, -1)
    
    # Mean
    mean = x_flatten.mean(axis=1, keepdims=True)
    
    # Variance
    variance = x_flatten.var(axis=1, keepdims=True)
    
    # Normalize
    x_normalized = (x_flatten - mean) / np.sqrt(variance + eps)
    
    out_reshaped = x_normalized.reshape(N, G, C // G, H, W)
    out = out_reshaped.reshape(N, C, H, W)
    
    # Scale and Shift
    out = gamma * out + beta
    
    cache = (x, x_normalized, mean, variance, gamma, beta, G, eps, x_flatten)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (1, C, 1, 1)
    - dbeta: Gradient with respect to shift parameter, of shape (1, C, 1, 1)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    x, x_normalized, mean, variance, gamma, beta, G, eps, x_flatten = cache
    N, C, H, W = dout.shape

    dbeta = np.sum(dout, axis=(0, 2, 3), keepdims=True) # (1, C, 1, 1)
    dgamma = np.sum(dout * x_normalized.reshape(N, C, H, W), axis=(0, 2, 3), keepdims=True) # (1, C, 1, 1)
    
    dout_reshaped = dout.reshape(N, G, C // G, H, W).reshape(N * G, -1)
    
    dout_norm_before_gamma = dout_reshaped * gamma.reshape(1, C).repeat(N * H * W, axis=0).reshape(N*G, -1)
    
    D_group = C // G * H * W
    sqrt_var = np.sqrt(variance + eps)
    
    # gradient (1/sqrt(var))
    dsqrt_var = np.sum(dout_norm_before_gamma * (x_flatten - mean), axis=1, keepdims=True)
    dvariance = dsqrt_var * (-0.5) * (variance + eps)**(-1.5)
    
    dmean_subtracted_term = dout_norm_before_gamma / sqrt_var
    dmean_sq = dvariance * 2.0 / D_group * (x_flatten - mean)
    
    dmean = np.sum(-(dmean_subtracted_term + dmean_sq), axis=1, keepdims=True)
    dx_flatten = dmean_subtracted_term + dmean_sq + dmean / D_group
    
    # (N*G, D_group) -> (N, G, C/G, H, W) -> (N, C, H, W)
    dx_reshaped = dx_flatten.reshape(N, G, C // G, H, W)
    dx = dx_reshaped.reshape(N, C, H, W)

    dgamma = dgamma.reshape(1, C, 1, 1)
    dbeta = dbeta.reshape(1, C, 1, 1)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta
