import numpy as np
def computeCost(X,y,theta):
    m = len(y)
    s = np.power((X.dot(theta)-np.transpose([y])),2)
    J = (1.0/(2*m))*s.sum(axis = 0)
    return J