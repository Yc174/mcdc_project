# Copyright UCL Business plc 2017. Patent Pending. All rights reserved.
#
# The MonoDepth Software is licensed under the terms of the UCLB ACP-A licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
#
# For any other use of the software not covered by the UCLB ACP-A Licence,
# please contact info@uclb.com

from __future__ import absolute_import, division, print_function

# only keep warnings and errors
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='0'

import numpy as np
import argparse
import re
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.misc
import matplotlib.pyplot as plt

import predictor.monodepth._init_paths

from monodepth_model import *
from monodepth_dataloader import *
from average_gradients import *
from ..base_predictor import BasePredictor
from tools.filter import Exponentially_weighted_averages

PREDICTOR = {'citys': 'model_cityscapes', 'kitti' : 'model_kitti', 'city2kitti':'model_city2kitti',}

def parse_args():

    parser = argparse.ArgumentParser(description='Monodepth TensorFlow implementation.')

    parser.add_argument('--encoder',          type=str,   help='type of encoder, vgg or resnet50', default='vgg')
    parser.add_argument('--image_path',       type=str,   help='path to the image', required=True)
    parser.add_argument('--checkpoint_path',  type=str,   help='path to a specific checkpoint to load', required=True)
    parser.add_argument('--input_height',     type=int,   help='input height', default=256)
    parser.add_argument('--input_width',      type=int,   help='input width', default=512)

    args = parser.parse_args()
    return args

class MonodepthPredictor(BasePredictor):
    def __init__(self, args):
        BasePredictor.__init__(self, args)
        self.input_height = args.input_height
        self.input_width = args.input_width
        self.predictor_model = args.predictor_model
        # self.image_path = args.image_path
        self.params = monodepth_parameters(
                        encoder=args.encoder,
                        height=args.input_height,
                        width=args.input_width,
                        batch_size=2,
                        num_threads=1,
                        num_epochs=1,
                        do_stereo=False,
                        wrap_mode="border",
                        use_deconv=False,
                        alpha_image_loss=0,
                        disp_gradient_loss_weight=0,
                        lr_loss_weight=0,
                        full_summary=False)
        self.model = self.build_model(self.params)

    def build_model(self, params):
        # self.left = tf.placeholder(tf.float32, [2, self.input_height, self.input_width, 3])
        # model = MonodepthModel(params, "test", self.left, None)

        # SESSION
        config = tf.ConfigProto(allow_soft_placement=True)
        g2 = tf.Graph()
        self.sess2 = tf.Session(config=config, graph=g2)

        with self.sess2.as_default():
            with g2.as_default():
                self.left = tf.placeholder(tf.float32, [2, self.input_height, self.input_width, 3])
                model = MonodepthModel(params, "test", self.left, None)

                # SAVER
                train_saver = tf.train.Saver()

                # INIT
                self.sess2.run(tf.global_variables_initializer())
                self.sess2.run(tf.local_variables_initializer())
                coordinator = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=self.sess2, coord=coordinator)

                # RESTORE
                # restore_path = args.checkpoint_path.split(".")[0]

                cur_file = os.path.dirname(__file__)
                tfmodel = os.path.join(cur_file,  'models', PREDICTOR[self.predictor_model])
                print('Loaded detector network {:s}'.format(tfmodel))

                if not os.path.isfile(tfmodel + '.meta'):
                    raise IOError(('{:s} not found.\nDid you download the proper networks from '
                                   'our server and place them properly?').format(tfmodel + '.meta'))

                train_saver.restore(self.sess2, tfmodel)

        return model

    def predict(self, bbox, time, fid, img=None):
        print('img shape: ', img.shape)
        img_height, img_width, channel = img.shape
        bbox = bbox[:4]
        bbox = np.array([bbox[1],bbox[0],bbox[3],bbox[2]])
        print('bbox before(float32) h0 w0 h1 w1:', bbox)

        hight0 = int(bbox[0]/img_height*self.input_height)
        width0 = int(bbox[1]/img_width*self.input_width)
        hight1 = int(bbox[2]/img_height*self.input_height)
        width1 = int(bbox[3]/img_width*self.input_width)

        print('bbox after(int32) h0 w0 h1 w1:', hight0, width0, hight1, width1)

        if len(self.times) == 0:
            vx = 0.
            self.pre_vx = 0.
            self.pre_bbox = np.array(bbox)
            time_diff = 0x3ffffff
        else:
            time_diff = time - self.times[-1]
        self.times.append(time)

        # bbox = Exponentially_weighted_averages(self.pre_bbox, bbox, fid, theta=0.8)

        disp_pp, depth_map, disp_img = self.depth_prediction(img)
        print("depth_map shape :", depth_map.shape)

        depth_bbox = depth_map[hight0:hight1, width0:width1]
        depth_bbox_shape = depth_bbox.shape
        depth_bbox = depth_bbox[int(depth_bbox_shape[0]/3):int(depth_bbox_shape[0]*2/3),
                     int(depth_bbox_shape[1]/3):int(depth_bbox_shape[1]*2/3)].astype(np.int32)

        # u, counts = np.unique(depth_bbox, return_counts=True)
        # x = u[counts.argmax()]
        #
        # if len(self.times) == 0:
        #     self.pre_x = x
        #
        # x = Exponentially_weighted_averages(self.pre_x, x, fid, theta=0.8)

        x = np.average(depth_bbox)
        x = Exponentially_weighted_averages(self.pre_x, x, fid, theta=0.9)
        # plt.figure("lena")
        # arr = depth_bbox.flatten()
        # n, bins, patches = plt.hist(arr, bins = 256, normed=0, facecolor='green', alpha=0.75)
        # plt.show()


        if len(self.result) > 0:
            # self.pre_x = self.result[-1]['x']
            vx = (x - self.pre_x) / 0.04

        vx = Exponentially_weighted_averages(self.pre_vx, vx, fid, theta=0.9)

        self.pre_bbox = bbox
        self.pre_x = x
        self.pre_vx = vx

        cur_pred = {
            'fid': fid,
            'vx': vx,
            'x': x,
            #             'ref_bbox': {
            #                 'top': float(bbox[0]), 'left': float(bbox[1]),
            #                 'bot': float(bbox[2]), 'right': float(bbox[3])
            #             }
        }

        self.result.append(cur_pred)
        #for show the depth map of the img
        depth_img = scipy.misc.imresize(disp_pp, [self.input_height, self.input_width])
        depth_img_bbox = depth_img[hight0:hight1, width0:width1]
        return cur_pred, depth_img, depth_img_bbox

    def depth_prediction(self, input_image):
        original_height, original_width, num_channels = input_image.shape
        input_image = scipy.misc.imresize(input_image, [self.input_height, self.input_width], interp='lanczos')
        input_image = input_image.astype(np.float32) / 255
        input_images = np.stack((input_image, np.fliplr(input_image)), 0)

        disp = self.sess2.run(self.model.disp_left_est[0], feed_dict={self.left: input_images})
        disp_pp = self.post_process_disparity(disp.squeeze()).astype(np.float32)

        # output_directory = os.path.join(os.path.dirname(__file__), '../..', 'show')
        # output_name = os.path.splitext(os.path.basename(self.image_path))[0]

        # np.save(os.path.join(output_directory, "{}_disp.npy".format(output_name)), disp_pp)
        disp_to_img = scipy.misc.imresize(disp_pp.squeeze(), [original_height, original_width])

        # depth_map = 2.262 * 0.22 / disp_pp
        depth_map =  0.7215377 * 0.54 / disp_pp #kitti
        # import pdb
        # pdb.set_trace()

        # plt.imsave(os.path.join(output_directory, "{}_disp.png".format(output_name)), disp_to_img, cmap='plasma')
        return  disp_pp, depth_map, disp_to_img

    def post_process_disparity(self, disp):
        _, h, w = disp.shape
        l_disp = disp[0, :, :]
        r_disp = np.fliplr(disp[1, :, :])
        m_disp = 0.5 * (l_disp + r_disp)
        l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
        l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
        r_mask = np.fliplr(l_mask)
        return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

# def main(_):
#     args = parse_args()
#     params = monodepth_parameters(
#         encoder=args.encoder,
#         height=args.input_height,
#         width=args.input_width,
#         batch_size=2,
#         num_threads=1,
#         num_epochs=1,
#         do_stereo=False,
#         wrap_mode="border",
#         use_deconv=False,
#         alpha_image_loss=0,
#         disp_gradient_loss_weight=0,
#         lr_loss_weight=0,
#         full_summary=False)
#
#     test_simple(args, params, img)

if __name__ == '__main__':
    tf.app.run()
