from __future__ import print_function
import numpy as np
import pickle


from tools.area_for_distance import AreaDistance_math_model, AreaDistance_NN
from tools.filter import Exponentially_weighted_averages
from tools.bird_view_projection import read_cam_param, bird_view_proj
from base_predictor import BasePredictor

from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


class RegressionPredictor(BasePredictor):

    def __init__(self, args):
        BasePredictor.__init__(self, args)
        with open('./x_reg.pickle', 'rb') as f:
            self.x_reg = pickle.load(f)
        with open('./vx_reg.pickle', 'rb') as f:
            self.vx_reg = pickle.load(f)

    def proj(self, u, v):
        x, y = bird_view_proj(u, v,
                              self.K, self.dist_coeff, self.R, self.T)
        return x, y

    def extract_bbox(self, b):
        '''
        "top": 597.832580566406,
        "right": 880.870239257812,
        "bot": 739.836364746094,
        "left": 686.165344238281
        '''

        #         h = b['bot'] - b['top'] + 1.
        #         w = b['right'] - b['left'] + 1.
        h = b[2] - b[0] + 1.
        w = b[3] - b[1] + 1.

        h /= 1200.
        w /= 1920.

        area = h * w
        area_dao = 1. / area
        x, y = self.proj((b[0] + b[2]) / 2, b[3])

        # return [h, w, area, x]
        return [h, w, area, x]

        # return [area_dao]

    def extract_sample(self, real_file, gt_file, time_file):
        with open(gt_file) as fin:
            gts = json.loads(fin.read())['frame_data']
        with open(real_file) as fin:
            rel = json.loads(fin.read())['frame_data']
        t_samples = [extract_bbox(e['ref_bbox']) for e in rel]
        # t_targets = [e['vx'] for e in gts] # e['x'] for dis, e['vx'] for relative v
        t_targets = [e['x'] for e in gts]  # e['x'] for dis, e['vx'] for relative v

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
                tvx.append((t_samples[i][-1] - t_samples[i - 1][-1]) / (times[i] - times[i - 1]))

        tvx[0] = tvx[1]
        for i in range(len(times)):
            t_samples[i].append(tvx[i])

        return t_samples, t_targets

    def predict(self, bbox, time, fid):
        bbox = bbox[:4]

        if len(self.times) == 0:
            vx = 0.
            self.pre_vx = 0.
            self.pre_bbox = np.array(bbox)
            time_diff = 0x3ffffff
        else:
            time_diff = time - self.times[-1]
        self.times.append(time)

        bbox = Exponentially_weighted_averages(self.pre_bbox, bbox, fid, theta=0.8)


        sample = self.extract_bbox(bbox)

        x = self.x_reg.predict([sample])[0]
        vx = self.vx_reg.predict([sample])[0]

#         print(sample, x, vx)
#         if len(self.result) > 0:
#             # self.pre_x = self.result[-1]['x']
#             vx = (x - self.pre_x) / time_diff

#         vx = Exponentially_weighted_averages(self.pre_vx, vx, fid, theta=0.9)

        self.pre_bbox = bbox
        self.pre_x = x
        self.pre_vx = vx

        cur_pred = {
            'fid': fid,
            'vx': vx,
            'x': x,
            #             'ref_bbox': {
            #                 'left': float(bbox[0]), 'top': float(bbox[1]),
            #                 'right': float(bbox[2]), 'bot': float(bbox[3])
            #             }
        }

        self.result.append(cur_pred)

        return cur_pred

