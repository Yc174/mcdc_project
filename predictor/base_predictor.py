# coding=utf-8
from __future__ import print_function
import json
import numpy as np
from tools.bird_view_projection import read_cam_param, bird_view_proj
from tools.filter import Exponentially_weighted_averages
from tools.draw_line import draw_one_time, draw_x_vx
from tools.error_estimation import x_err_esti, vx_err_esti, high_esti


class BasePredictor:
    def __init__(self, args):
        self.result = []
        self.times = []
        self.pre_bbox = np.zeros(4).astype(np.float32)
        self.pre_vx = 0.
        self.pre_x = 0.
        self.x_sum = 0.
        # read camera parameters
        with open(args.cam_calib) as f:
            cam_param = json.load(f)
            self.K, self.dist_coeff, self.R, self.T = read_cam_param(cam_param)

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

        x, y = bird_view_proj((bbox[0] + bbox[2]) / 2, bbox[3],
                              self.K, self.dist_coeff, self.R, self.T)
        x = np.sqrt(x * x - 3.8 ** 2)

        if len(self.result) > 0:
            # self.pre_x = self.result[-1]['x']
            vx = (x - self.pre_x) / time_diff

        # if len(self.result) == 1:
        #     self.pre_vx = vx

        vx = Exponentially_weighted_averages(self.pre_vx, vx, fid, theta=0.9)

        self.pre_bbox = bbox
        self.pre_x = x
        self.pre_vx = vx

        cur_pred = {
            'fid': fid,
            'vx': vx,
            'x': x,
            'ref_bbox': {
                'left': float(bbox[0]), 'top': float(bbox[1]),
                'right': float(bbox[2]), 'bot': float(bbox[3])
            }
        }

        self.result.append(cur_pred)

        return cur_pred

    def to_json(self, filename):
        print('Save prediction to ', filename)

        if len(self.result) > 1:
            self.result[0]['vx'] = self.result[1]['vx']
            self.result[0]['x'] = self.result[1]['x']

        with open(filename, 'w') as fout:
            data = {'frame_data': self.result}
            json.dump(data, fout, indent=4, ensure_ascii=False)

    def draw_valid_line(self, gt_file, before):
        result = self.result
        fids = [r['fid'] for r in result]
        vx = [r['vx'] for r in result]
        x = [r['x'] for r in result]

        with open(gt_file) as f:
            ground_truth = json.load(f)
            gt_x = [gt['x'] for gt in ground_truth["frame_data"]]
            gt_vx = [gt['vx'] for gt in ground_truth["frame_data"]]
        # before = 10
        draw_one_time(fids[:before], x[:before], vx[:before], gt_x[:before], gt_vx[:before], gt_file.split('/')[-1])
        # draw_one_time(fids[:500],x[:500],vx[:500],gt_x[:500],gt_vx[:500],gt_file.split('/')[-1])
        
    def draw_test_line(self, name):
        result = self.result
        fids = [r['fid'] for r in result]
        vx = [r['vx'] for r in result]
        x = [r['x'] for r in result]
        draw_x_vx(fids, x, vx, name.split('/')[-1])

    def err_estimation(self, gt_file, before):
        result = self.result
        fids = [r['fid'] for r in result]
        vx = [r['vx'] for r in result]
        x = [r['x'] for r in result]

        with open(gt_file) as f:
            ground_truth = json.load(f)
            gt_x = [gt['x'] for gt in ground_truth["frame_data"]]
            gt_vx = [gt['vx'] for gt in ground_truth["frame_data"]]
        # before = 10
        err_x = x_err_esti(x[:before], gt_x[:before])
        err_vx = vx_err_esti(vx[:before], gt_vx[:before])
        high = high_esti(x[:before], gt_x[:before])
        print('Error for x :', err_x)
        print('Error for vx:', err_vx)
        print('high        :', high)

