import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    scores = X.dot(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    for i in range(X.shape[0]):
        loss += np.log(np.exp(scores[i]).sum()) - scores[i, y[i]]
        for c in range(dW.shape[1]):
            dW[:, c] += X[i] * np.exp(scores[i, c]) / np.exp(scores[i]).sum()
        dW[:, y[i]] -= X[i]
    loss /= X.shape[0]
    dW /= X.shape[0]
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    scores = X.dot(W)
    loss = sum(np.log(np.exp(scores).sum(axis=1)) - scores[np.arange(len(y)), y]) / X.shape[0] + reg * np.sum(W * W)
    scores_exp = np.exp(scores)
    scores_exp /= scores_exp.sum(axis=1)[:, np.newaxis]
    scores_exp[np.arange(len(y)), y] -= 1
    dW = np.transpose(X).dot(scores_exp)
    dW /= X.shape[0]
    dW += 2 * reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
