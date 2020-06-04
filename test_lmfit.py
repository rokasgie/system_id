from lmfit import minimize, Parameters
import numpy as np
from sklearn.metrics import mean_squared_error


def mse(y_pred, y_true):
    # tot_loss = 0
    # for y in y_true.T:
        # tot_loss += mean_squared_error(y, y_pred)
    # return tot_loss / len(y_true)
    return mean_squared_error(y_true.T, y_pred.T)


def getFirst(a, b, x):
    return a*x+b


def getSecond(a, b, x):
    return a*x+b


def model(parameters, xs, ys):
    a1 = parameters['a1']
    b1 = parameters['b1']
    a2 = parameters['a2']
    b2 = parameters['b2']

    y1_pred = getFirst(a1, b1, xs)
    y2_pred = getSecond(a2, b2, xs)

    pred = np.vstack([y1_pred, y2_pred])
    true = np.vstack([ys[0], ys[1]])
    loss = pred - true
    return loss


if __name__ == "__main__":
    params = Parameters()
    params.add('a1', value=1)
    params.add('b1', value=-1)
    params.add('a2', value=3)
    params.add('b2', value=1.5)

    xs = np.array(list(range(5)))
    y1s = np.array([2*x-1 for x in xs])
    y2s = np.array([0.5*x+1 for x in xs])
    ys = np.vstack((y1s, y2s))
    model(params, xs, ys)
    result = minimize(model, params, args=(xs, ys))
    result.params.pretty_print()
