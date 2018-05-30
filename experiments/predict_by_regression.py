#coding=utf-8
#简单写了个用h,w,area预测x, vx的拟合， 误差在0.007, 0.25
from __future__ import print_function
import glob
import json
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
# import tensorflow as tf

from tools.bird_view_projection import read_cam_param, bird_view_proj
from tools.filter import Exponentially_weighted_averages
from tools.error_estimation import x_err_esti, vx_err_esti

class BirdProj():
    def __init__(self, cam_calib):
        # read camera parameters
        with open(cam_calib) as f:
            cam_param = json.load(f)
            self.K, self.dist_coeff, self.R, self.T = read_cam_param(cam_param)

    def proj(self, u, v):
        x, y = bird_view_proj(u, v,
                       self.K, self.dist_coeff, self.R, self.T)
        return x, y

bird_proj = BirdProj('valid_02/camera_parameter.json')

def extract_bbox(b):
    '''
    "top": 597.832580566406,
    "right": 880.870239257812,
    "bot": 739.836364746094,
    "left": 686.165344238281
    '''

    h = b['bot'] - b['top']+1.
    w = b['right']- b['left']+1.

    h /= 1200.
    w /= 1920.

    area = h * w
    area_dao = 1. / area
    # x, y = bird_proj.proj((b['left'] + b['right'])/2, b['bot'])

    x, y = bird_proj.proj((b['top'] + b['bot'])/2, b['right'])

    # return [h, w, area, x]
    return [h, w, area, x]

    # return [area_dao]

def extract_sample(real_file, gt_file, time_file):
    with open(gt_file) as fin:
        gts = json.loads(fin.read())['frame_data']
    with open(real_file) as fin:
        rel = json.loads(fin.read())['frame_data']
    t_samples = [extract_bbox(e['ref_bbox']) for e in rel]
    # t_targets = [e['vx'] for e in gts] # e['x'] for dis, e['vx'] for relative v
    t_targets = [e['vx'] for e in gts]  # e['x'] for dis, e['vx'] for relative v

    times = []
    with open(time_file, 'r') as fin:
        for line in fin.readlines():
            times.append(float(line))

    # add vx as new feature
    tvx = []
    for i in range(len(times)):
        if i == 0:
            tvx.append(0)
        else:
            tvx.append((t_samples[i][-1] - t_samples[i-1][-1]) / (times[i] - times[i - 1]))

    tvx[0] = tvx[1]
    for i in range(len(times)):
        t_samples[i].append(tvx[i])

    return t_samples, t_targets

def try_different_method(clf, x_train, y_train, x_test, y_test):
    clf.fit(x_train,y_train)

    score = clf.score(x_test, y_test)
    print('score: ', score)

    result = clf.predict(x_test)

    plt.figure()
    plt.plot(np.arange(len(result)), y_test,'go-',label='true value')
    plt.plot(np.arange(len(result)),result,'ro-',label='predict value')
    plt.title('score: %f'%score)
    plt.legend()
    plt.show()

    # 保存Model
    with open('./clf.pickle', 'wb') as f:
        pickle.dump(clf, f)

    # 读取Model
    with open('./clf.pickle', 'rb') as f:
        clf2 = pickle.load(f)
        # 测试读取后的Model
        print(clf2.predict(x_test[0:1]), y_test[0:1])

    print('relative x error: ', x_err_esti(result, y_test))
    print('mean absolute error: ', vx_err_esti(result, y_test))
    result = clf2.predict([x_test[0]])
    print(x_test[0])
    print(result)


def train(t_smples, t_targets):

    from sklearn import tree
    # for x
    reg = tree.DecisionTreeRegressor(random_state=1, min_samples_split=310, max_depth=4)
    #
    # # from sklearn import linear_model
    # # reg = linear_model.LinearRegression()
    #
    # # from sklearn import svm
    # # reg = svm.SVR()
    #
    # # from sklearn import neighbors
    # # reg = neighbors.KNeighborsRegressor()
    #
    # try_different_method(reg, samples, targets, samples, targets)
    x_train, x_test, y_train, y_test = train_test_split(t_smples, t_targets, test_size=0.1, random_state=42)
    try_different_method(reg, x_train, y_train, x_test, y_test)


def test(x_test, y_test):
    # 读取Model
    with open('./clf.pickle', 'rb') as f:
        clf = pickle.load(f)
        # 测试读取后的Model
        print(clf.predict(x_test[0:1]), y_test[0:1])

    result = clf.predict(x_test)
    # print('relative x error: ', x_err_esti(result, y_test))
    print('mean absolute error: ', vx_err_esti(result, y_test))

def train_model(clf, t_smples, t_targets):
    x_train, x_test, y_train, y_test = train_test_split(t_smples, t_targets, test_size=0.1, random_state=42)
    clf.fit(x_train, y_train)
    result = clf.predict(x_test)
    # print('Train relative x error: ', x_err_esti(result, y_test))
    print('Train mean absolute error: ', vx_err_esti(result, y_test))
    return clf

def test_model(clf, x_test, y_test):
    result = clf.predict(x_test)
    # print('Test  relative x error: ', x_err_esti(result, y_test))
    print('Test  mean absolute error: ', vx_err_esti(result, y_test))


def train_and_valid(x_train, y_train, x_test, y_test, t_max_depth=3, t_min_samples_split=180):
    from sklearn import tree
    reg = tree.DecisionTreeRegressor(random_state=1,
                                     min_samples_split = t_min_samples_split,
                                     max_depth = t_max_depth)
    clf = train_model(reg, x_train, y_train)
    test_model(clf, x_test, y_test)


if __name__ == '__main__':
    valid_gt_files = glob.glob('./valid_pre/*gt.json')
    x_train = []
    y_train = []
    for file in valid_gt_files[:-1]:
        # print(file)
        time_file = file[:-7] + 'time.txt'
        real_file = file[:-7] + 'pre.json'
        t_samples, t_targets = extract_sample(real_file, file, time_file)
        x_train += t_samples
        y_train += t_targets


    x_test = []
    y_test = []
    for file in valid_gt_files[-1:]:
        # print(file)
        time_file = file[:-7] + 'time.txt'
        real_file = file[:-7] + 'pre.json'
        t_samples, t_targets = extract_sample(real_file, file, time_file)
        x_test += t_samples
        y_test += t_targets

    train(x_train, y_train)
    test(x_test, y_test)

    # for max_depth in range(1, 6):
    #     for min_samples_split in range(100, 400, 10):
    #         print('=======', max_depth, min_samples_split)
    #         train_and_valid(x_train, y_train, x_test, y_test, max_depth, min_samples_split)