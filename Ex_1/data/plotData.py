from matplotlib import pyplot as plt
from matplotlib.pyplot import figure


def plotData(x,y):
    plt.plot(x,y,'rx',markersize = 10,label = 'training Data')
    plt.xlabel('Polulation of the city')
    plt.ylabel('Profit')
    plt.show()