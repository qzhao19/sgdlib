
import numpy as np

from sgdlib import tg
from sgdlib import sgd
from sgdlib import sag
from sgdlib import scd
from sgdlib import lbfgs
from sgdlib.loss import log_loss
from sgdlib.common import load_data
from sgdlib.common import Regularizer, StepDecay

def test_sgd(X, y):
    w = np.random.rand(X.shape[1], 1)
    # w = np.ones((X.shape[1], 1))
    optimizer = sgd.SGD(
        x0 = w,
        loss_func = log_loss.LogLoss(), 
        lr_decay = StepDecay(lr = 0.1), 
        regularizer = Regularizer(penalty = None, alpha = 0.0)
    )
    opt_w = optimizer.optimize(X, y)
    return opt_w

def test_sag(X, y):
    w = np.random.rand(X.shape[1], 1)
    # w = np.ones((X.shape[1], 1))
    optimizer = sag.SAG(
        x0 = w,
        loss_func = log_loss.LogLoss(), 
        lr_decay = StepDecay(lr = 0.1), 
        regularizer = Regularizer(penalty = None, alpha = 0.0)
    )
    opt_w = optimizer.optimize(X, y)
    return opt_w

def test_tg(X, y):
    w = np.random.rand(X.shape[1], 1)
    # w = np.ones((X.shape[1], 1))
    optimizer = tg.TruncatedGradient(
        x0 = w,
        loss_func = log_loss.LogLoss(), 
        lr_decay = StepDecay(lr = 0.1), 
        regularizer = Regularizer(penalty = None, alpha = 0.0)
    )
    opt_w = optimizer.optimize(X, y)
    return opt_w


def test_scd(X, y):
    # w = np.random.rand(X.shape[1], 1)
    # w = np.ones((X.shape[1], 1))
    w = np.zeros((X.shape[1]))
    optimizer = scd.SCD(
        x0 = w,
        loss_func = log_loss.LogLoss()
    )
    opt_w = optimizer.optimize(X, y)
    return opt_w


def test_lbfgs(X, y):
    x0 = np.zeros(X.shape[1])
    linesearch_params = {
        "decrease_factor": 0.5, 
        "increase_factor": 2.1,
        "ftol": 1e-4,
        "max_linesearch": 40,
        "condition": "WOLFE",
        "wolfe": 0.9,
        "max_step": 1e+20,
        "min_step": 1e-20
    }
    optimizer = lbfgs.LBFGS(x0,
        linesearch_params = linesearch_params,
        loss_func = log_loss.LogLoss()
    )
    opt_w = optimizer.optimize(X, y)
    return opt_w


def main():
    X, y = load_data("./data/ionosphere.data", sep = ",")

    print("SGD")
    opt_w = test_sgd(X, y)
    print(opt_w)

    print("SAG")
    opt_w = test_sag(X, y)
    print(opt_w)

    print("Truncated Gradient")
    opt_w = test_tg(X, y)
    print(opt_w)

    print("SCD")
    opt_w = test_scd(X, y)
    print(opt_w)

    print("lbfgs")
    opt_w = test_lbfgs(X, y)
    print(opt_w)

if __name__ == '__main__':
    main()