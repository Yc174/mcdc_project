import numpy as np
from NN_py import forward_propagation
parameters = {
    'W1': [[ 1.6489735, 2.5770068,  1.3255011,  1.5453409, -1.7536916, -2.1271243, 3.5497422, -1.1388757,  1.4115595, -1.240432 ]],
    'b1': [[-2.3562818, -2.1307294, -1.9104774, -5.0135794, -2.2701466, -1.8844584, -1.8767567, -2.7430067, -3.9331024, -2.3172555]],
    'W2': [[3.4655418],
           [3.4225938],
           [3.4731112],
           [3.8253808],
           [0.13520782],
           [-0.2229167],
           [3.2454066],
           [0.39814854],
           [3.5073137],
           [0.34731576]],
    'b2' : [[1.9144869]],

}


def find_area(bbox):
    #bbox.shape=(5,)
    return (bbox[2]-bbox[0]+1.)*(bbox[3]-bbox[1]+1.)

def AreaDistance_math_model(bbox, k, b):
    #dis=k/a+b
    area = find_area(bbox)
    if not area == 0.:
        return k/area+b
    else:
        return -1

def AreaDistance_NN(bbox):
    area = find_area(bbox)
    x = np.asarray(100000 / area).reshape(1, 1)
    dis = forward_propagation(x, parameters)
    #print(dis.shape)
    return dis[0][0]

