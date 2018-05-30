
import json
import cv2
import numpy as np
import argparse
import math

from tools.bird_view_projection import read_cam_param, bird_view_proj
# from detector.darknet.yolov3_detector import YoloV3Detector


def draw_bbox_with_rec(frame, bboxes):
    for b in bboxes:
        left, top, right, bottom = [int(e) for e in b[:4]]
        cv2.rectangle(frame, (left, top), (right, bottom), color = [0, 0, 255], thickness=10)

def show_bbox(frame, bboxes):
    draw_bbox_with_rec(frame, bboxes)

    frame = cv2.resize(frame, (400, 400))

    cv2.imshow('image', frame)

    k = cv2.waitKey(0)

    if k == 27:  # wait for ESC key to exit
        cv2.destroyAllWindows()
    elif k == ord('s'):  # wait for 's' key to save and exit
        cv2.imwrite('messigray.png', frame)
        cv2.destroyAllWindows()

class CarSelector():
    def __init__(self, args):
        with open(args.cam_calib) as f:
            cam_param = json.load(f)
            self.K, self.dist_coeff, self.R, self.T = read_cam_param(cam_param)

    def _get_line(self, bbox):
        left = bbox[0]
        top = bbox[1]
        right = bbox[2]
        bot = bbox[3]
        x2, y2 = bird_view_proj(left, bot,
                              self.K, self.dist_coeff, self.R, self.T)
        # print('left-bot: ', left, bot, x, y2)
        x1, y1 = bird_view_proj(right, bot,
                              self.K, self.dist_coeff, self.R, self.T)
        # print('right-bot: ', right, bot, x, y1)
        return [y1, y2, min(x1, x2)]

    def select_front_car(self, bboxes):
        front_line = self._get_line([100., 1200., 1700., 1200.])

        bot_lines = np.array([self._get_line(b) for b in bboxes])
        print("front_line: ", front_line)
        
        print(bot_lines)
        inter_l = np.maximum(front_line[0], bot_lines[:, 0])
        inter_r = np.minimum(front_line[1], bot_lines[:, 1])
        length = bot_lines[:, 1] - bot_lines[:, 0]
        # print(inter_l)
        # print(inter_r)
        overlap = np.maximum(0.0, inter_r - inter_l)
        ratio = overlap / length
        print('overlap: ', overlap)
        print('length: ', length)
        print('ratio: ', ratio)
        
        r_sort = ratio.argsort()[::-1]
        
        i = r_sort[0]
        bbox = bboxes[i]
        
        for s in r_sort[1:]:
#             if math.fabs(ratio[s] - ratio[i]) <= 0:
            if ratio[s] >= 0.5:
                if bot_lines[s][2] < bot_lines[i][2]:
                    bbox = bboxes[s]
                    i = s
            else: 
                break

        return bbox 

    def find_middle_car_use_iou(self, boxes, rang=None):
        if rang is None:
            rang = [500, 500, 1000, 1000]
            rang = np.array(rang)
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)

        rang_x1 = rang[0]
        rang_y1 = rang[1]
        rang_x2 = rang[2]
        rang_y2 = rang[3]

        rang_area = (rang_y2 - rang_y1 + 1) * (rang_x2 - rang_x1 + 1)

        xx1 = np.maximum(rang_x1, boxes[:, 0])
        yy1 = np.maximum(rang_y1, boxes[:, 1])
        xx2 = np.minimum(rang_x2, boxes[:, 2])
        yy2 = np.minimum(rang_y2, boxes[:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        ovr = inter / (areas + rang_area - inter)
        inds = ovr.argsort()[::-1]

        return boxes[inds[0]]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'car selector')
    parser.add_argument('-c', '--cam-calib', default='../valid/camera_parameter.json',
                        help='calibrated camera parameter file path')
    args = parser.parse_args()

    detector = YoloV3Detector()

    # imgs = ['290.jpg', 'test02-single_bbox_0.jpg', 'test06-single_bbox_0.jpg',
    #         'test15-single_bbox_0.jpg', 'valid15-single_bbox_0.jpg', 'valid17-single_bbox_0.jpg']
    # imgs = ['valid15-single_bbox_0.jpg']
    imgs = ['valid17-single_bbox_0.jpg']
    # image_file = '/Users/hzshuai/MCDC/mcdc_data/err_img/test15-single_bbox_0.jpg'
    image_path = '/Users/hzshuai/MCDC/mcdc_data/err_img/'
    for file in imgs:
        print("process: ", file)
        image_file = image_path + file
        frame = cv2.imread(image_file)

        dets = detector.detect(frame)
        # print(dets)
        # print('[')
        # for d in dets:
        #     print('[' + ', '.join('%.2lf' % e for e in d) + '],')
        # print(']')

#         dets = [
# [1074.34, 600.03, 1248.29, 723.42, 0.99],
# [970.48, 626.53, 1048.31, 678.76, 0.98],
# [873.10, 627.27, 913.12, 665.61, 0.89],
# [694.25, 630.93, 742.03, 672.40, 0.84],
# [595.10, 626.86, 700.62, 702.96, 0.82],
# ]

        selector = CarSelector(args)
        bbox = selector.select_front_car(dets)

        show_bbox(frame, dets)
        # show_bbox(frame, [bbox])

