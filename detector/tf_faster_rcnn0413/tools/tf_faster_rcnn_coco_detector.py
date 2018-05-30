#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#import _init_paths
import detector.tf_faster_rcnn0413.tools._init_paths
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms

from utils.timer import Timer
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import glob
import argparse

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1

VOC_CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

COCO_CLASSES = ('__background__',
                'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
                'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
                'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
                'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
                'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',),
        'res101': ('res101_faster_rcnn_iter_110000.ckpt','res101_faster_rcnn_iter_1190000.ckpt')}
DATASETS= {'pascal_voc': ('voc_2007_trainval',),'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',),
           'coco':('coco_2014_train+coco_2014_valminusminival',)}

def find_middle_car_use_area(boxes,rang=None):
    if rang==None:
        rang=[[500,500,1000,1000]]
        rang=np.array(rang)

    xx1=np.maximum(rang[0,0],boxes[:,0])
    yy1=np.maximum(rang[0,1],boxes[:,1])
    xx2=np.minimum(rang[0,2],boxes[:,2])
    yy2=np.minimum(rang[0,3],boxes[:,3])

    w = np.maximum(0.0, xx2 - xx1 + 1)
    h = np.maximum(0.0, yy2 - yy1 + 1)
    inter = w * h
    inds = inter.argsort()[::-1]

    return boxes[inds[0]]


def find_middle_car_use_iou(boxes, rang=None):
    if rang == None:
        rang = [[500, 500, 1000, 1000]]
        rang = np.array(rang)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    rang_x1=rang[0,0]
    rang_y1=rang[0,1]
    rang_x2=rang[0,2]
    rang_y2=rang[0,3]
    rang_area=(rang_y2-rang_y1+1)*(rang_x2-rang_x1+1)

    xx1 = np.maximum(rang_x1, boxes[:, 0])
    yy1 = np.maximum(rang_y1, boxes[:, 1])
    xx2 = np.minimum(rang_x2, boxes[:, 2])
    yy2 = np.minimum(rang_y2, boxes[:, 3])
    w = np.maximum(0.0, xx2 - xx1 + 1)
    h = np.maximum(0.0, yy2 - yy1 + 1)
    inter = w * h

    ovr=inter/(areas+rang_area-inter)
    inds = ovr.argsort()[::-1]

    return boxes[inds[0]]

def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    plt.show()
    
def detections_thresh(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    return dets[inds,:]

def process(sess, net, im, only_car=True, vis=False):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
#     im_file = os.path.join(cfg.DATA_DIR, 'mcdc/anc105_j03_ldl_u8794_01_8_3', image_name)
#     im = cv2.imread(im_file)
    

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
#     print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.5
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(COCO_CLASSES[1:]):
        if only_car :
            if cls=='car':
                cls_ind += 1 # because we skipped background
                cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
                cls_scores = scores[:, cls_ind]
                dets = np.hstack((cls_boxes,
                                  cls_scores[:, np.newaxis])).astype(np.float32)
                keep = nms(dets, NMS_THRESH)
                dets = dets[keep, :]
                if vis:
                    vis_detections(im, cls, dets, thresh=CONF_THRESH)
                else:
#                     print('*'*30+'car'+'*'*30)
                    #pass
#                     print('before dets: ', dets)
                    dets = detections_thresh(im, cls, dets, thresh=CONF_THRESH)
#                     print('after dets: ', dets)
#                     try:
#                         dets = find_middle_car_use_iou(dets)
#                     except:
#                         print("no find car")
#                         dets[0]
#                         dets = find_middle_car_use_area(dets)

                    dets = dets.reshape(-1,5)
                    # print(dets)
                    #vis_detections(im, cls, dets, thresh=CONF_THRESH)
        else:
            cls_ind += 1 # because we skipped background
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                              cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]
            if vis:
                vis_detections(im, cls, dets, thresh=CONF_THRESH)
            else:
                pass
            dets=detections_thresh(im, cls, dets, thresh=CONF_THRESH)
#     print('return: ', dets)
    return dets       

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--detector_net', dest='detector_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='res101')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712 coco]',
                        choices=DATASETS.keys(), default='coco')
    args = parser.parse_args()

    return args

class TfFasterRcnnDetector:
    def __init__(self, args):
        #load model
        # model path
        demonet = args.detector_net
        dataset = args.dataset
        cur_file=os.path.dirname(__file__)
        tfmodel = os.path.join(cur_file,'..','output', demonet, DATASETS[dataset][0], 'default',
                                  NETS[demonet][1])
        print('Loaded detector network {:s}'.format(tfmodel))
        
        if not os.path.isfile(tfmodel + '.meta'):
            raise IOError(('{:s} not found.\nDid you download the proper networks from '
                           'our server and place them properly?').format(tfmodel + '.meta'))
        # set config
        tfconfig = tf.ConfigProto(allow_soft_placement=True)
        tfconfig.gpu_options.allow_growth = True
        g1 = tf.Graph()
        # init session
        self.sess1 = tf.Session(config=tfconfig, graph=g1)

        with self.sess1.as_default():

            with g1.as_default():

                # load network
                if demonet == 'vgg16':
                    self.net = vgg16()
                elif demonet == 'res101':
                    self.net = resnetv1(num_layers=101)
                else:
                    raise NotImplementedError
                self.net.create_architecture("TEST", 81,
                                      tag='default', anchor_scales=[4, 8, 16, 32])
                saver = tf.train.Saver()
                saver.restore(self.sess1, tfmodel)
    
    def detect(self, img):
        dets = process(self.sess1, self.net, img)
        # return dets[0]
        return dets

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()
    
    detector = TfFasterRcnnDetector(args)
    

    
    filename='/data/mcdc_data/train/train_images/*/*.jpg'
    im_files = glob.glob(filename)
#     print(filename)
#     print(im_files)
    for img in im_files:
        print(img)
    for im_name in im_files:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Demo for data/demo/{}'.format(im_name))
        img = cv2.imread(im_name)
        dets = detector.detect(img)
        print(dets)
        
#     plt.show()

