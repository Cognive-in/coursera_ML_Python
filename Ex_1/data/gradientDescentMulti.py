import numpy as np
import computeCost as cc

def gradientDescentMulti(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = np.zeros((num_iters, 1))

    for i in range(num_iters):
        theta = theta - alpha * (1.0 / m) * np.transpose(X).dot(X.dot(theta) - np.transpose([y]))
        J_history[i] = cc.computeCost(X, y, theta)

    return theta,J_history