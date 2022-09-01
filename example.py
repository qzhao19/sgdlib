
from sgdlib import sgd
from sgdlib.loss import log_loss
from sgdlib.utils import load_data
from sgdlib.utils import Regularizer, StepDecay

def main():
    X, y = load_data("./data/ionosphere.data", sep = ",")

    optimizer = sgd.SGD(loss = log_loss.LogLoss(), 
        lr_decay = StepDecay(lr = 0.1), 
        regularizer = Regularizer(penalty = None, alpha = 0.0)
    )
    opt_w = optimizer.optimize(X, y)

    print(opt_w)

if __name__ == '__main__':
    main()