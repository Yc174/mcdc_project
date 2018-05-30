import numpy as np
from tools.area_for_distance import AreaDistance_math_model, AreaDistance_NN
from tools.filter import Exponentially_weighted_averages
from base_predictor import BasePredictor

class AreaPredictor(BasePredictor):
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

        #x = AreaDistance_math_model(bbox, 100000., 0.)
        x = AreaDistance_NN(bbox)
        # x = np.sqrt(x * x - 3.8 ** 2)

        if len(self.result) > 0:
            # self.pre_x = self.result[-1]['x']
            vx = (x - self.pre_x) / time_diff

        vx = Exponentially_weighted_averages(self.pre_vx, vx, fid, theta=0.9)

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

