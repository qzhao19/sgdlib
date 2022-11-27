
import numpy as np

from sgdlib import sgd
from sgdlib import sag
from sgdlib import scd
from sgdlib import lbfgs
from sgdlib.loss import log_loss
from sgdlib.utils import load_data
from sgdlib.utils import Regularizer, StepDecay


def test_sgd(X, y):
    optimizer = sgd.SGD(
        loss_func = log_loss.LogLoss(), 
        lr_decay = StepDecay(lr = 0.1), 
        regularizer = Regularizer(penalty = None, alpha = 0.0)
    )
    opt_w = optimizer.optimize(X, y)
    return opt_w

def test_sag(X, y):
    optimizer = sag.SAG(
        loss_func = log_loss.LogLoss(), 
        lr_decay = StepDecay(lr = 0.1), 
        regularizer = Regularizer(penalty = None, alpha = 0.0)
    )
    opt_w = optimizer.optimize(X, y)
    return opt_w

def test_scd(X, y):
    optimizer = scd.SCD(
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

    test_lbfgs(X, y)
    

if __name__ == '__main__':
    main()