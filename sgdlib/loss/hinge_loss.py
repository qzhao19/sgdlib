import numpy as np

class HingeLoss(object):
    def __init__(self, threshold):
        self.threshold = threshold

    def gradient(self, X, y, beta):
        """
        This function computes the gradient of the squared hinge loss
        objective function
        :param beta: A dx1 vector of beta values
        :param lambdat: Value of the regularization parameter
        :param X: A dxn matrix of features
        :param y: A nx1 vector of labels
        :return: A dx1 vector containing the gradient
        """
        return -2/len(y) * (np.maximum(0, self.threshold - ((y[:, np.newaxis]*X).dot(beta)))).dot(y[:, np.newaxis]*X)

    def evaluate(self, X, y, beta):
        """
        This function computes the value of the objective function for the squared
        hinge loss. It is vectorized so as to enable faster computation.
        :param beta: A dx1 vector of beta values
        :param lambdat: Value of the regularization parameter
        :param X: A dxn matrix of features
        :param y: A nx1 vector of labels
        :return: A float value, equivalent to the value of the objective function
        """
        return 1/len(y) * (np.sum(
            (np.maximum(0, self.threshold-((y[:, np.newaxis]*X).dot(beta)))**2)))