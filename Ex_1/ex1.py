import data.warmUpExercise as a
import data.computeCost as cc
import data.gradientDescent as gd
import data.plotData as pd
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

#============Part 1=================
print('Running warmUpExercise')
print('5x5 Identity Matrix')
print(a.warm())

#==========Part 2===================
print('Plotting Data')
data = np.loadtxt('ex1data1.txt',delimiter =",")
X = data[:,0]
y = data[:,1]
m = len(y)

pd.plotData(X,y)

#================part 3====================
print('Gradient Descent')
X_padded = np.column_stack((np.ones((m,1)),X))
theta = np.zeros((2,1))

iterations = 1500
alpha = 0.01

print('testing functions')
print(cc.computeCost(X_padded,y,theta))
theta = gd.gradientDescent(X_padded,y,theta,alpha,iterations)
print("{:f}, {:f}".format(theta[0,0], theta[1,0]))

#plot linear fit
plt.plot(X,X_padded.dot(theta),'-',label = 'Linear regression')
plt.legend(loc = 'lower right')
plt.draw()


#predicts values for population size of 35000 and 70000
predict1 = np.array([1,3.5]).dot(theta)
print("For population = 35000 we predict a profit of {:f}".format(float(predict1*10000)))
predict2 = np.array([1, 7]).dot(theta)
print('For population = 70,000, we predict a profit of {:f}'.format( float(predict2*10000) ))

#================ part 4============================
print("j(theta_0,theta1)")
theta0_vals = np.linspace(-10,10,100)
theta1_vals = np.linspace(-1,4,100)

J_vals = np.zeros((len(theta0_vals),len(theta1_vals)))
for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        t = [[theta0_vals[i]],[theta1_vals[j]]]
        J_vals[i,j] = cc.computeCost(X_padded,y,t)

J_vals = np.transpose(J_vals)
fig = plt.figure()
ax = fig.gca(projection='3d')
theta0_vals, theta1_vals = np.meshgrid(theta0_vals, theta1_vals) # necessary for 3D graph
surf = ax.plot_surface(theta0_vals, theta1_vals, J_vals, cmap=cm.coolwarm, rstride=2, cstride=2)
fig.colorbar(surf)
plt.xlabel('theta_0')
plt.ylabel('theta_1')

fig = plt.figure()
ax = fig.add_subplot(111)

cset = plt.contour(theta0_vals, theta1_vals, J_vals, np.logspace(-2, 3, 20), cmap=cm.coolwarm)
fig.colorbar(cset)
plt.xlabel('theta_0')
plt.ylabel('theta_1')
plt.plot(theta[0,0], theta[1,0], 'rx', markersize=10, linewidth=2)
plt.show()