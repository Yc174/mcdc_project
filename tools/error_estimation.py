import numpy as np

def x_err_esti(x, gt_x):
    x = np.array(x)
    gt_x = np.array(gt_x)
    if len(x.shape) == 2:
        n = x.shape[0] * x.shape[1]
        return np.sum(np.fabs(x-gt_x)/gt_x)/n
    elif len(x.shape) == 3:
        n = x.shape[0] * x.shape[1] * x.shape[2]
        return np.sum(np.fabs(x-gt_x)/gt_x)/n
    else:
        return np.sum(np.fabs(x - gt_x) / gt_x) / x.shape[0]


def vx_err_esti(vx, gt_vx):
    vx = np.array(vx)
    gt_vx = np.array(gt_vx)
    if len(vx.shape) == 2:
        n = vx.shape[0]*vx.shape[1]
        return np.sum(np.fabs(vx-gt_vx))/n
    elif len(vx.shape) == 3:
        n = vx.shape[0] * vx.shape[1]*vx.shape[2]
        return np.sum(np.fabs(vx - gt_vx)) /n
    else:
        return np.sum(np.fabs(vx - gt_vx)) / vx.shape[0]

def high_esti(x, gt_x):
    x = np.array(x)
    gt_x = np.array(gt_x)
    high = np.sum(np.sqrt(np.maximum(0.,x*x - gt_x*gt_x))) / x.shape[0]
    return high