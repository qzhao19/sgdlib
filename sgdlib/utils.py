import numpy as np
import math

def load_data(file_path, sep = '\t'):
    """Download dating data to testing alogrithm

    Args:
        file_name: input file path which allows to read a txt file
    Returns:
        retuen_mat: a matrix of data
        label_vect: a vectro conatins labels 
    """
    data = []
    label = []
    with open(file_path, 'r') as file:
        contents = [line.strip().split(sep) for line in file.readlines()]

    for i in range(len(contents)):
        data.append(contents[i][:-1])
        label.append(contents[i][-1])

    return np.array(data, dtype=float), np.array(label, dtype=float).reshape(-1)


class Regularizer(object):
    def __init__(self, penalty = None, alpha = 0.0):
        self.penalty = penalty
        self.alpha = alpha

    def evaluate(self, W, num_samples):
        if not self.penalty:
            reg = 0.0
        elif self.penalty == "L1":
            reg = np.sum(np.abs(W))
        elif self.penalty == "L2":
            reg = np.dot(W.T, W)
        else:
            raise ValueError("No type")
        return self.alpha * reg / (2 * num_samples)

    def gradient(self, W, num_samples):
        if not self.penalty:
            grad = 0.0
        elif self.penalty == "L1":
            grad = self.alpha * np.sign(W) 
        elif self.penalty == "L2":
            grad = self.alpha * W 
        else:
            raise ValueError("No type")
        return grad / num_samples


class StepDecay(object):
    def __init__(self, lr, gamma = 0.5, step_size = 10):
        self.base_lr = lr
        self.gamma = gamma
        self.step_size = step_size

    def compute(self, epoch):
        lr = self.base_lr * math.pow(self.gamma,  
                math.floor((1 + epoch) / self.step_size))
        return lr