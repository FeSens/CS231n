import numpy as np
from random import shuffle

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
  #dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  d = np.zeros(W.shape)

  for i in xrange(num_train):
    scores = X[i].dot(W)
    dS = np.zeros((10))
    f = scores
    f -= np.max(scores) # f becomes [-666, -333, 0]
    p = np.exp(f) / np.sum(np.exp(f)) # safe to do, gives the correct answer
    #p = 1.0 / (1 + np.exp(-f))
    correct_class_score = p[y[i]]
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
    loss += -np.log(correct_class_score/np.sum(p))
    dS = p
    dS[y[i]] -= 1 
    d += X[i][np.newaxis].T.dot(dS[np.newaxis])
  loss /= num_train
  d /= num_train
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  d += reg*W
  dW = d
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_classes = W.shape[1]
  num_train = X.shape[0]
  d = np.zeros(W.shape)
  p = np.zeros((num_train, num_classes))

  scores = X.dot(W)
  f = scores
  f[np.arange(num_train)] -= np.max(scores[np.arange(num_train)])
  fexp = np.exp(f)
  p[np.arange(num_train)] = fexp / np.sum(fexp, axis=1, keepdims=True)
  
  score_cc = -np.log(p[range(num_train),y])
  loss = np.sum(score_cc)
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  dP = p
  dP[range(num_train),y] += -1
 
  dW = X.T.dot(dP)/num_train
  dW += W*reg
  
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

