
#coding=utf-8

from __future__ import print_function
import os
import glob
import json
import numpy as np
import pickle
import argparse
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
# import keras as K
# from keras.models import Sequential
# from keras.layers import SimpleRNN, Activation, Dense,Bidirectional,LSTM,TimeDistributed
# from keras.optimizers import Adam

from tools.bird_view_projection import read_cam_param, bird_view_proj
from tools.filter import Exponentially_weighted_averages
from tools.error_estimation import x_err_esti, vx_err_esti
#import tools.nn as nn

HIGHT = 500.
WIDTH = 500.

def parse_args():
    parser = argparse.ArgumentParser(description='Demo for experiment_lstm, copyright@Ready Player One')
    parser.add_argument('--saved_model', #required=True,
                        help='directory for valid or test video', type=str, default = 'models/')

    parser.add_argument('--x_vx_mode', type = str, help="Trained dataset ['x','vx', 'x_vx']",
                        default='vx')
    # parser.add_argument('--bbox_height', type=float, help='bbox_height', default=500.)
    # parser.add_argument('--bbox_width', type=float, help='bbox_width', default=500.)
    parser.add_argument('--batch_size', type=int, help='batch_size', default=100)
    parser.add_argument('--n_steps', type=int, help='n_steps', default=80)
    parser.add_argument('--learning_rate', type=float, help='learning_rate', default=0.006)
    parser.add_argument('--use_gt', type=bool, help='use_gt', default=False)
    parser.add_argument('--is_training', type=bool, help='is_training', default=True)
    args = parser.parse_args()
    return args

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

bird_proj = BirdProj('/data/mcdc_data/valid/camera_parameter.json')

def extract_bbox(b):
    '''
    "top": 597.832580566406,
    "right": 880.870239257812,
    "bot": 739.836364746094,
    "left": 686.165344238281
    '''

    h = (b['bot'] - b['top']+1.) / WIDTH
    w = (b['right']- b['left']+1.) / HIGHT
    area = h * w
    x, y = bird_proj.proj((b['left'] + b['right'])/2, b['bot'])

    # return [h, w, area, x]
    # return [h, w, area, area_dao]
    return [h, w, area, x, y]

def extract_sample(filename, time_file, pre_file, use_gt = True):
    with open(filename) as fin:
        gts = json.loads(fin.read())['frame_data']
    with open(pre_file) as fin:
        pres = json.loads(fin.read())['frame_data']
    if use_gt:
        t_samples = [extract_bbox(e['ref_bbox']) for e in gts]
    else:
        t_samples = [extract_bbox(e['ref_bbox']) for e in pres]

    # t_targets = [e['vx'] for e in gts] # e['x'] for dis, e['vx'] for relative v
    t_targets = [[e['x'], e['vx']] for e in gts]  # e['x'] for dis, e['vx'] for relative v

    times = []
    with open(time_file, 'r') as fin:
        for line in fin.readlines():
            times.append(float(line))
            
    delt_time = []
    for i in range(len(times)):
        if i == 0:
            delt_time.append(0)
        else:
            delt_time.append(times[i] - times[i-1])
    delt_time[0] = delt_time[1]    

#     for i in range(len(times)):
#         t_samples[i].append(delt_time[i])

    return t_samples, t_targets

def err(prediction,gt_x):
    prediction = np.asarray(prediction, np.float32)
    gt_x = np.asarray(gt_x, np.float32)
    return x_err_esti(prediction, gt_x)


def get_train_batch(X_train, y_train, batch_size, Tx):
    num_video, every_long, n_x = X_train.shape
    n_y = y_train.shape[-1]
    X_train_batch = np.zeros((batch_size,Tx,n_x), dtype=np.float32)
    y_train_batch = np.zeros((batch_size,Tx,n_y), dtype=np.float32)

    for batch in range(batch_size):
        which_video = np.random.choice(num_video, 1)[0]
        inds = np.random.choice(every_long-Tx, 1)[0]
        X_train_batch[batch,:,:] = X_train[which_video,inds:(inds+Tx),:]
        y_train_batch[batch,:,:] = y_train[which_video,inds:(inds+Tx),:]

    # inds = np.random.choice(num_video, batch_size)
    # X_train_batch = X_train[inds,:,:]
    # y_train_batch = y_train[inds,:,:]
    # if batch_size == 1:
    #     X_train_batch = X_train_batch.reshape(batch_size, Tx,n_x)
    #     y_train_batch = y_train_batch.reshape(batch_size, Tx,n_y)
    return X_train_batch, y_train_batch

def get_test_batch(X_test,y_test,batch_size,inds):
    num_video, every_long, n_x = X_test.shape
    n_y = y_test.shape[-1]
    # inds = np.random.choice(num_video, 1)[0]
    X_test_batch = np.tile(X_test[inds,:,:], (batch_size,1)).reshape(batch_size, every_long,n_x)
    y_test_batch = np.tile(y_test[inds,:,:], (batch_size,1)).reshape(batch_size, every_long,n_y)
    return X_test_batch, y_test_batch

def draw_line(xs, pred, labels, name):
    plt.ion()
    plt.show()
    # plt.plot(xs, X_train_batch[0, :,0].flatten(),xs, X_train_batch[0, :,1].flatten(),xs, X_train_batch[0, :,2].flatten())
    plt.plot(xs, pred[0, :, 0].flatten(), 'b', xs, labels[0, :, 0].flatten(), 'r')
    # plt.ylim((-2, 20))
#     plt.draw()
#     plt.pause(1)
    plt.savefig(name)
    plt.close()

def draw_subplot(x,y,y_label='vx',line='target',):
    if line=='target':
        #draw real line
        plt.plot(x,y,color='red',label=line)
        # plt.xlabel('time-axis label')
        plt.ylabel(y_label)
        # plt.title(y_label+' line plot')
        # plt.show()
    elif line=='sample':
        #draw ground truth
        plt.plot(x,y,color='blue',label=line)
        # plt.xlabel('time-axis label')
        plt.ylabel(y_label)
        # plt.title(y_label+' line plot')
        # plt.show()
    else:
        pass
    plt.legend(loc='upper right')

def draw_x_vx(time,x,vx,gt_x,gt_vx,name):
    plt.figure(name)
    plt.subplot(211)
    draw_subplot(time, x, y_label='x', line='sample' )
    draw_subplot(time, gt_x, y_label='x', line='target')
    plt.title(name)
    plt.subplot(212)
    draw_subplot(time,vx,y_label='vx',line='sample')
    draw_subplot(time,gt_vx,y_label='vx',line='target')
    plt.savefig('show/'+name)
#     plt.draw()
#     plt.pause(1)
    plt.close()

class LSTMRNN(object):
    def __init__(self, n_steps, input_size, output_size, cell_size, batch_size, LR, x_vx_mode):
        self.n_steps = n_steps
        self.input_size = input_size
        self.output_size = output_size
        self.cell_size = cell_size
        self.batch_size = batch_size
        self.learning_rate = LR
        self.x_vx_mode = x_vx_mode
        with tf.name_scope('inputs'):
            self.xs = tf.placeholder(tf.float32, [batch_size, None, input_size], name='xs')
            self.ys = tf.placeholder(tf.float32, [batch_size, None, output_size], name='ys')
        with tf.variable_scope('in_hidden'):
            self.add_input_layer()
        with tf.variable_scope('LSTM_cell'):
            self.add_cell()
        with tf.variable_scope('out_hidden'):
            self.add_output_layer()
        with tf.name_scope('cost'):
            self.compute_cost()
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

    def add_input_layer(self,):
        l_in_x = tf.reshape(self.xs, [-1, self.input_size], name='2_2D')  # (batch*n_step, in_size)
        # Ws (in_size, cell_size)
        Ws_in = self._weight_variable([self.input_size, self.cell_size])
        # bs (cell_size, )
        bs_in = self._bias_variable([self.cell_size,])
        # l_in_y = (batch * n_steps, cell_size)
        with tf.name_scope('Wx_plus_b'):
            l_in_y = tf.tanh(tf.matmul(l_in_x, Ws_in) + bs_in)
        # reshape l_in_y ==> (batch, n_steps, cell_size)
        self.l_in_y = tf.reshape(l_in_y, [self.batch_size, -1, self.cell_size], name='2_3D')

    def add_cell(self):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.cell_size, forget_bias=1.0, state_is_tuple=True)

        # lstm_cell = tf.contrib.rnn.MultiRNNCell(
        #     [lstm_cell() for _ in range(3)], state_is_tuple=True)

        with tf.name_scope('initial_state'):
            self.cell_init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
        self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(
            lstm_cell, self.l_in_y, initial_state=self.cell_init_state, time_major=False)


    def add_output_layer(self):
        # shape = (batch * steps, cell_size)
        l_out_x = tf.reshape(self.cell_outputs, [-1, self.cell_size], name='2_2D')
        Ws_out = self._weight_variable([self.cell_size, self.output_size])
        bs_out = self._bias_variable([self.output_size, ])
        # shape = (batch * steps, output_size)
        with tf.name_scope('Wx_plus_b'):
            outputs = tf.matmul(l_out_x, Ws_out) + bs_out
            self.pred = tf.reshape(outputs, [self.batch_size, -1, self.output_size])
    def compute_cost(self):
        if self.x_vx_mode == 'x' or self.x_vx_mode == 'vx':
            losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
                [tf.reshape(self.pred, [-1], name='reshape_pred')],
                [tf.reshape(self.ys, [-1], name='reshape_target')],
                [tf.ones([self.batch_size * self.n_steps * self.output_size], dtype=tf.float32)],
                average_across_timesteps=True,
                softmax_loss_function=self.ms_error,
                name='losses'
            )
        elif self.x_vx_mode == 'x_vx':
            losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
                        [tf.reshape(self.pred[:,:,0], [-1], name='reshape_pred')],
                        [tf.reshape(self.ys[:,:,0], [-1], name='reshape_target')],
                        [tf.ones([self.batch_size * self.n_steps ], dtype=tf.float32)],
                        average_across_timesteps=True,
                        softmax_loss_function=self.ms_error,
                        name='losses'
                    )+tf.contrib.legacy_seq2seq.sequence_loss_by_example(
                        [tf.reshape(self.pred[:,:,1], [-1], name='reshape_pred')],
                        [tf.reshape(self.ys[:,:,1], [-1], name='reshape_target')],
                        [tf.ones([self.batch_size * self.n_steps], dtype=tf.float32)],
                        average_across_timesteps=True,
                        softmax_loss_function=self.ms_error,
                        name='losses'
                    )
        else:
            pass

        with tf.name_scope('average_cost'):
            self.cost = tf.div(
                tf.reduce_sum(losses, name='losses_sum'),
                self.batch_size,
                name='average_cost')
            tf.summary.scalar('cost', self.cost)

    @staticmethod
    def ms_error(labels, logits):
        # return tf.reduce_sum(tf.square(tf.subtract(labels, logits)))
        return tf.square(tf.subtract(labels, logits))

    def _weight_variable(self, shape, name='weights'):
        initializer = tf.random_normal_initializer(mean=0., stddev=1.,)
        return tf.get_variable(shape=shape, initializer=initializer, name=name)

    def _bias_variable(self, shape, name='biases'):
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable(name=name, shape=shape, initializer=initializer)

if __name__ == '__main__':
    args = parse_args()
    print(args)

    is_training = args.is_training
    use_gt = args.use_gt
    M = args.batch_size
    Tx = args.n_steps
    lr  = args.learning_rate
    x_vx_mode = args.x_vx_mode
    saved_model = args.saved_model
    saved_model_path = os.path.join(saved_model, x_vx_mode,x_vx_mode+'_lstm')
    print('saved_model_path:', saved_model_path)

    # n_a = 100
    n_a = 16

    train_slice = 0.7

    valid_gt_files = glob.glob('./valid_pre/*gt.json')
    samples = []
    targets = []
    np.random.seed(1337)
    for file in valid_gt_files:
#         if int(file[-10:-8])>14:
#             continue
        print(file)
        time_file = file[:-7] + 'time.txt'
        pre_file = file[:-7] + 'pre.json'
        t_samples, t_targets = extract_sample(file, time_file, pre_file, use_gt)
        samples.append(t_samples)
        targets.append(t_targets)

    # print(samples[0], targets[0])
    print(len(samples), len(targets))

    samples = np.asarray(samples, np.float32)
    targets = np.asarray(targets, np.float32)
    xs = np.arange(500)
    i = 0
    for tar in targets:
        plt.ion()
        plt.show()
        # plt.plot(xs, X_train_batch[0, :,0].flatten(),xs, X_train_batch[0, :,1].flatten(),xs, X_train_batch[0, :,2].flatten())
        plt.plot(xs, tar[:, 0].flatten(), 'r', xs, tar[:, 1].flatten(), 'b')
#         plt.ylim((-2, 20))
    #     plt.draw()
    #     plt.pause(1)
        plt.savefig('show/gt_show/%s.png'%valid_gt_files[i].split('/')[-1])
        i = i+1
        plt.close()
    
    print(samples.shape, targets.shape)

    if x_vx_mode == 'x':
        n_x = 5
        n_y = 1
        targets = targets[:,:,0][:,:,np.newaxis]
    elif x_vx_mode == 'vx':
        n_x = 5
        n_y = 1
        targets = targets[:,:,1][:,:,np.newaxis]
    elif x_vx_mode == 'x_vx':
        n_x = 5
        n_y = 2
        targets = targets[:, :, :]
    else:
        pass

    LR  = tf.Variable(lr,trainable=False)

    inds = np.arange(samples.shape[0])
    np.random.shuffle(inds)

    train_inds = inds[:int(len(inds)*train_slice)]
    if is_training:
        test_inds  = inds[int(len(inds)*train_slice):]
    else:
        test_inds = inds
    X_train = samples[train_inds,:,:]
    y_train = targets[train_inds,:,:]
    X_test  = samples[test_inds, :,:]
    y_test  = targets[test_inds, :,:]

    valid_gt_file_names = [f.split('/')[-1] for f in valid_gt_files]
    test_flies = np.asarray(valid_gt_file_names)[test_inds]
    print("total test file:", test_flies)

    model = LSTMRNN(Tx, n_x, n_y, n_a, M, LR, x_vx_mode)
    saver = tf.train.Saver()
    sess = tf.Session()

    if is_training == True:

        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("logs", sess.graph)
        # tf.initialize_all_variables() no long valid from
        # 2017-03-02 if using tensorflow >= 0.12
        if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
            init = tf.initialize_all_variables()
        else:
            init = tf.global_variables_initializer()
        sess.run(init)

        for i in range(20001):
            X_train_batch, y_train_batch = get_train_batch(X_train,y_train,M,Tx)

            feed_dict = {
                model.xs: X_train_batch,
                model.ys: y_train_batch,
                # model.cell_init_state: state  # use last state as the initial state for this run
            }

            _, cost, state, pred = sess.run(
                [model.train_op, model.cost, model.cell_final_state, model.pred],
                feed_dict=feed_dict)

            if i % 20 == 0:
                print(i, 'cost: ', round(cost, 4))
                result = sess.run(merged, feed_dict)
                writer.add_summary(result, i)

            if i % 1000 ==0:
                save_path = saver.save(sess, saved_model_path+'_%d'%i)
                print("Save to path: ", save_path)
                num_video_test = X_test.shape[0]
                # ind = np.random.choice(num_video_test,1)[0]
                for ind in range(num_video_test):
                    X_test_batch, y_test_batch = get_test_batch(X_test, y_test, M, ind)
                    print("test video :",test_flies[ind])

                    feed_dict = {
                        model.xs: X_test_batch,
                        # model.cell_init_state: state  # use last state as the initial state for this run
                    }
                    y_test_pred, outputs = sess.run([model.pred, model.cell_outputs],feed_dict=feed_dict)
                    # print('*'*50+'outputs'+'*'*50)
                    # print(outputs)
                    xs = np.arange(500)
                    if x_vx_mode == 'x':
                        print('error for x:', x_err_esti(y_test_pred[:,:,0], y_test_batch[:,:,0]))
                        draw_line(xs, y_test_pred, y_test_batch,name = 'show/curve_train/x_%s.png'%test_flies[ind][:-7])
                    elif x_vx_mode == 'vx':
                        print('error for vx:', vx_err_esti(y_test_pred[:, :, 0], y_test_batch[:, :, 0]))
                        draw_line(xs, y_test_pred, y_test_batch, name='show/curve_train/vx_%s.png'%test_flies[ind][:-7])
                    elif x_vx_mode == 'x_vx':
                        print('error for x:', x_err_esti(y_test_pred[:, :, 0], y_test_batch[:, :, 0]))
                        print('error for vx:', vx_err_esti(y_test_pred[:, :, 1], y_test_batch[:, :, 1]))
                        draw_x_vx(xs, y_test_pred[0, :, 0].flatten(), y_test_pred[0, :, 1].flatten(),
                                  y_test_batch[0,:,0].flatten(),y_test_batch[0,:,1].flatten(),name = "curve_train/x_vx_%s.png"%test_flies[ind][:-7])

            if i % 10000 == 0 and i != 0 :
                lr = lr *0.1
                print('lr:', lr)
                sess.run(tf.assign(model.learning_rate, lr))

    else:

        saver.restore(sess, saved_model_path+'_20000')
        num_video_test = X_test.shape[0]
        for ind in range(num_video_test):
            X_test_batch, y_test_batch = get_test_batch(X_test, y_test, M, ind)
            print("test video :", test_flies[ind])

            feed_dict = {
                model.xs: X_test_batch,
                # model.cell_init_state: state  # use last state as the initial state for this run
            }
            y_test_pred, outputs = sess.run([model.pred, model.cell_outputs], feed_dict=feed_dict)
            xs = np.arange(500)
            if x_vx_mode == 'x':
                print('error for x:', x_err_esti(y_test_pred[:, :, 0], y_test_batch[:, :, 0]))
                draw_line(xs, y_test_pred, y_test_batch, name='show/curve_test/x_%s_test.png' % test_flies[ind][:-7])
            elif x_vx_mode == 'vx':
                print('error for vx:', vx_err_esti(y_test_pred[:, :, 0], y_test_batch[:, :, 0]))
                draw_line(xs, y_test_pred, y_test_batch, name='show/curve_test/vx_%s_test.png' %test_flies[ind][:-7])
            elif x_vx_mode == 'x_vx':
                print('error for x:', x_err_esti(y_test_pred[:, :, 0], y_test_batch[:, :, 0]))
                print('error for vx:', vx_err_esti(y_test_pred[:, :, 1], y_test_batch[:, :, 1]))
                draw_x_vx(xs, y_test_pred[0, :, 0].flatten(), y_test_pred[0, :, 1].flatten(),
                          y_test_batch[0, :, 0].flatten(), y_test_batch[0, :, 1].flatten(), 
                          name="curve_test/x_vx_%s_test.png"%test_flies[ind][:-7])

