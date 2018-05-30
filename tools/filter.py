import numpy as np
from draw_line import draw_subplot
import matplotlib.pyplot as plt
def Exponentially_weighted_averages(last, now_in, fid, theta=0.8):
    #equal to average 1/(1-theta) step
    if fid == 0:
        return now_in
    else:
        now = theta*last+(1-theta)*now_in
    return now

if __name__ == '__main__':
    # test draw subplot
    x=np.arange(-20,20,1)
    y1=(3*x*x-12)/2.
    y2 = (3 * x * x - 12) / 2.+np.random.randn(len(x))*30

    # y2=[for ]
    # draw_line(x,y1,line='real')
    draw_subplot(x,y2)
    plt.show()