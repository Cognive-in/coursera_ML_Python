import numpy as np
import data.computeCost as cc

def gradientDescent(X,y,theta,alpha,num_iters):
    m = len(y)
    J_hist = np.zeros((num_iters,1))
    for i in range(num_iters):
        theta = theta - alpha*(1.0/m)*np.transpose(X).dot(X.dot(theta)-np.transpose([y]))
        J_hist[i] = cc.computeCost(X,y,theta)
    return theta