from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class FullyConnectedNet(object):
    """Class for a multi-layer fully connected neural network.

    Network contains an arbitrary number of hidden layers, ReLU nonlinearities,
    and a softmax loss function. This will also implement dropout and batch/layer
    normalization as options. For a network with L layers, the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional and the {...} block is
    repeated L - 1 times.

    Learnable parameters are stored in the self.params dictionary and will be learned
    using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout_keep_ratio=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout_keep_ratio: Scalar between 0 and 1 giving dropout strength.
            If dropout_keep_ratio=1 then the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
            are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
            initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
            this datatype. float32 is faster but less accurate, so you should use
            float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers.
            This will make the dropout layers deteriminstic so we can gradient check the model.
        """
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################

        dims = [input_dim] + hidden_dims + [num_classes]

        for i in range(self.num_layers):
           num_layer_str = str(i+1)

           # weight and bias parameters
           self.params[f'W{num_layer_str}'] = weight_scale * np.random.randn(dims[i], dims[i+1])
           self.params[f'b{num_layer_str}'] = np.zeros(dims[i+1])

           # scale and shift parameters
           if (i < self.num_layers-1) and self.normalization is not None:
              self.params[f'gamma{num_layer_str}'] = np.ones(dims[i+1])
              self.params[f'beta{num_layer_str}'] = np.zeros(dims[i+1])

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype.
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """Compute loss and gradient for the fully connected net.
        
        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
            scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
            names to gradients of the loss with respect to those parameters.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        cache = []

        # From the 1st layer to the (L-1) layer --> {affine - batchnorm - relu}
        for i in range(1, self.num_layers):
            W = self.params[f'W{i}']
            b = self.params[f'b{i}']
           
            # affine - batchnorm - relu
            if self.normalization == "batchnorm":
                gamma = self.params[f'gamma{i}']
                beta = self.params[f'beta{i}']
                bn_param = self.bn_params[i-1]

                X, cache_i = affine_bn_relu_forward(X, W, b, gamma, beta, bn_param)
                cache.append(cache_i)
            elif self.normalization == "layernorm":
                gamma = self.params[f'gamma{i}']
                beta = self.params[f'beta{i}']
                ln_param = self.bn_params[i-1]
                
                # Since there is no helper function, manually call (affine -> layernorm -> relu)
                affine_out, fc_cache = affine_forward(X, W, b)
                norm_out, ln_cache = layernorm_forward(affine_out, gamma, beta, ln_param)
                X, relu_cache = relu_forward(norm_out)
                
                cache_i = (fc_cache, ln_cache, relu_cache)
                cache.append(cache_i)
            # affine - relu
            else:
                x_affine, cache_affine = affine_forward(X, W, b)
                x_relu, cache_relu = relu_forward(x_affine)

                X = x_relu
                cache.append((cache_affine, cache_relu))

            if self.use_dropout:
                X, dropout_cache = dropout_forward(X, self.dropout_param)
                cache.append(dropout_cache)

        # Last L layer
        L = self.num_layers
        WL = self.params[f'W{L}']
        bL = self.params[f'b{L}']

        scores, cache_L = affine_forward(X, WL, bL)
        cache.append(cache_L)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early.
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the   #
        # scale and shift parameters.                                              #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # caculate loss
        loss, dscores = softmax_loss(scores, y)
        # regularization
        for i in range(1, self.num_layers+1):
           W = self.params[f'W{i}']
           loss += 0.5 * self.reg * np.sum(W**2)
        
        # gradient
        cache_L = cache.pop()
        dh, dwL, dbL = affine_backward(dscores, cache_L)
        grads[f'W{L}'] = dwL + self.reg * self.params[f'W{L}']
        grads[f'b{L}'] = dbL
        
        for i in range(self.num_layers-1, 0, -1):
            if self.use_dropout:
                dropout_cache = cache.pop()
                dh = dropout_backward(dh, dropout_cache)
                
            cache_i = cache.pop()
            if self.normalization == "batchnorm":
                dh, dw, db, dgamma, dbeta = affine_bn_relu_backward(dh, cache_i)

                grads[f'gamma{i}'] = dgamma
                grads[f'beta{i}'] = db
            elif self.normalization == "layernorm":
                fc_cache, ln_cache, relu_cache = cache_i
                
                d_relu = relu_backward(dh, relu_cache)
                d_norm, dgamma, dbeta = layernorm_backward(d_relu, ln_cache)
                dh, dw, db = affine_backward(d_norm, fc_cache)
                
                grads[f'gamma{i}'] = dgamma
                grads[f'beta{i}'] = dbeta
            else:
                cache_affine, cache_relu = cache_i

                # ReLU backpropagation
                dh_relu = relu_backward(dh, cache_relu)
                # affine backpropagation
                dh, dw, db = affine_backward(dh_relu, cache_affine)

            # save gradient
            grads[f'W{i}'] = dw + self.reg * self.params[f'W{i}']
            grads[f'b{i}'] = db
        # Adding an L2 regularization term to the gradient
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads