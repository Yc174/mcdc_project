#coding=utf-8
from __future__ import print_function
import time
import argparse
from glob import glob
import os, cv2
import numpy as np
import matplotlib.pyplot as plt

from detector.base_detector import BaseDetector
# from detector.tf_faster_rcnn0413.tools.tf_faster_rcnn_detector import TfFasterRcnnDetector
# from detector.tf_faster_rcnn0413.tools.tf_faster_rcnn_mcdc_detector import TfFasterRcnnDetector
from detector.tf_faster_rcnn0413.tools.tf_faster_rcnn_coco_detector import TfFasterRcnnDetector
#from detector.darknet.yolov3_detector import YoloV3Detector

from predictor.base_predictor import BasePredictor
from predictor.test_predictor import TestBasePredictor
from predictor.monodepth.monodepth_simple_for_video import MonodepthPredictor
from predictor.lstm_predictor_x_vx import LSTMPredictor
from predictor.car_selector import CarSelector
#from predictor.area_predictor import AreaPredictor
from predictor.regression_predictor import RegressionPredictor

from tools.video_reader import VideoReader
from tools.choice_bbox import find_middle_car_use_iou, vis_detections

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',),
        'res101': ('res101_faster_rcnn_iter_110000.ckpt','res101_faster_rcnn_iter_1190000.ckpt')}
DATASETS= {'pascal_voc': ('voc_2007_trainval',),'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',),
           'coco':('coco_2014_train+coco_2014_valminusminival',),
           'mcdc':('mcdc_train_10000',)
          }
PREDICTOR = {'citys': 'model_cityscapes', 'kitti' : 'model_kitti', 'city2kitti':'model_city2kitti',}

def read_data(video, time_file):
    vreader = VideoReader(video)
    times = []
    with open(time_file, 'r') as fin:
        for line in fin.readlines():
            times.append(float(line))

    print('Total frames: ', len(times), times[0], times[-1])
    return vreader, times

def parse_args():
    parser = argparse.ArgumentParser(description='Demo for MCDC, copyright@Ready Player One')
    parser.add_argument('--input-dir', required=True,
                        help='directory for valid or test video', type=str)
    parser.add_argument('--output-dir', required=True,
                        help='directory for result', type=str)
    parser.add_argument('-c', '--cam-calib', required=True,
                        help='calibrated camera parameter file path')

    #for depth predictor
    parser.add_argument('--predictor_model', dest='predictor_model', help='Model to use [citys kitti]',
                        choices=PREDICTOR.keys(), default='city2kitti')
    # parser.add_argument('--image_path', type=str, help='path to the image', required=True)
    parser.add_argument('--encoder', type=str, help='type of encoder, vgg or resnet50', default='vgg')
    parser.add_argument('--input_height',     type=int,   help='input height', default=256)
    parser.add_argument('--input_width',      type=int,   help='input width', default=512)

    parser.add_argument('--detector', dest='detector',
                        help='tf-faster-rcnn', default='tf-faster-rcnn', type=str)
    parser.add_argument('--detector_net', dest='detector_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='res101')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712 coco mcdc]',
                        choices=DATASETS.keys(), default='coco')
    parser.add_argument('--x_vx_mode', type = str, help="mode for lstm predictor ['x','vx', 'x_vx']",
                        default='x_vx')
    parser.add_argument('--learning_rate', type=float, help='learning_rate', default=0.006)

    parser.add_argument('--lstm_predictor_model',
                        help='directory for lstm predictor models', type=str, default = 'models/')

    parser.add_argument('--data-type', help='val, test', type=str, default='test')

    parser.add_argument('--gpu', type=int, default=0,
                        help='choose one gpu for distribution computing')

    args = parser.parse_args()
    return args

def save_to_img(video, img, bbox, fid, depth_img=None, depth_bbox=None, pre=None):
    show_dir = 'show/%s' % os.path.basename(video)
    cv2.rectangle(img, 
                  (int(bbox[0]), int(bbox[1])),
                  (int(bbox[2]), int(bbox[3])),
                  (0, 255, 0), 3
                 )
    x = int((bbox[0] + bbox[2])/2)
    y = int(bbox[3])
    cv2.rectangle(img, 
                  (x, y),
                  (x, y),
                  (0, 0, 255), 10
                 )
    if pre is not None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        img = cv2.putText(img, 'Relative distance: %f m'%(pre['x']), (50, 50), font, 1.2, (255, 0, 0), 2)
        img = cv2.putText(img, 'Relative speed   : %f m/s' % (pre['vx']), (50, 100), font, 1.2, (255, 0, 0), 2)
    if not os.path.exists(show_dir):
        os.makedirs(show_dir)
    cv2.imwrite('%s/single_bbox_%d.jpg' % (show_dir, fid), img)
    if depth_img is not None:
        # cv2.imwrite('%s/%d_depth.png' % (show_dir, fid), depth_img)
        # cv2.imwrite('%s/%d_depth_bbox.png' % (show_dir, fid), depth_bbox)
        # depth_to_img = scipy.misc.imresize(depth_img, [256, 512])
        plt.imsave('%s/%d_depth.png' % (show_dir, fid), depth_img, cmap='plasma')
        # depth_bbox_to_img = scipy.misc.imresize(depth_bbox, [300, 400])
    if depth_bbox is not None:
        plt.imsave('%s/%d_depth_bbox.png' % (show_dir, fid), depth_bbox, cmap='plasma')

        # plt.imsave(os.path.join(output_directory, "{}_disp.png".format(output_name)), disp_to_img, cmap='plasma')

def main():
    args = parse_args()
    print(args)

    # 1. browse videos and time file in input dir
    videos = glob(os.path.relpath(args.input_dir) + '/*video*.avi')
    time_files = [v[:-4] + '_time.txt' for v in videos] # second
    gt_files = [v[:-4] + '_gt.json' for v in videos]

    # 2. init detector model
    if args.detector == 'tf-faster-rcnn':
        detector = TfFasterRcnnDetector(args)
#    elif args.detector == 'yolo-v3':
#        detector = YoloV3Detector(args)
    else:
        detector = BaseDetector(args)

    selector = CarSelector(args)

    # 3. process video one by one
    for video, time_file, gt_file in zip(videos, time_files, gt_files):
        
#        if int(video[-6:-4]) % 6 != args.gpu - 2:
#             continue
#         if int(video[-6:-4]) < 12:
#              continue
        print(video, time_file)
#         predictor = BasePredictor(args)
        # predictor = RegressionPredictor(args)
        predictor = MonodepthPredictor(args)
        # predictor = LSTMPredictor(args)

        vreader, times = read_data(video, time_file)

        before = 500
        start = time.time()
        last_bbox = np.zeros((1, 5), dtype=np.float32)

        for fid, t in enumerate(times):
            # print('fid: ', fid)
            frame = vreader.next()
            dets = detector.detect(frame)

            if fid == 0:
                # bbox = find_middle_car_use_iou(dets, rang=np.array([600, 500, 1160, 900]))
                bbox = selector.select_front_car(dets)
            else:
                bbox = selector.find_middle_car_use_iou(dets, rang=last_bbox)

            last_bbox = bbox

            #for MonodepthPredictor
            pre = predictor.predict(bbox, t, fid, frame)
            cur_pred, depth_img, depth_img_bbox = pre

            #for others
            # pre = predictor.predict(bbox, t, fid, frame)

            if fid % 1 == 0:
                save_to_img(video, frame, bbox, fid, depth_img, depth_img_bbox, cur_pred)
                vis_detections(frame, 'car', dets, thresh=0.5, video=video, fid=fid)

            if fid >= before:
                break

        print('>> Time elapsed: %lf s' % (time.time() - start))

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
            print('makedirs ', args.output_dir)
        result_file = os.path.join(args.output_dir, os.path.basename(video)[:-4] + '_pre.json')
        predictor.to_json(result_file)


        # visual for debug
        if args.data_type == 'test':
            predictor.draw_test_line(video)
            pass
        else:
            predictor.draw_valid_line(gt_file, fid)
            predictor.err_estimation(gt_file, fid)


if __name__ == '__main__':
    main()

# CUDA_VISIBLE_DEVICES=0 python main.py --input-dir=./test --output-dir=./test_pre --cam-calib=./test/camera_parameter.json
